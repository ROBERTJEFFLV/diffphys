#!/usr/bin/env python3
"""Replay saved long EVAL flights; no policy inference or training imports.

Export with the training Python, then render with any Python containing NumPy
and Pygame. All failed flights stop at their first failure, before frozen padding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def select_scenes(summary, completed_count=13, *, allow_fewer=False):
    failed=sorted((r for r in summary['scenes'] if r['failure_reason']=='arena_exit'),key=lambda r:r['scene_id'])
    completed=sorted((r for r in summary['scenes'] if r['completed']),key=lambda r:(r['mass_kg'],r['scene_id']))
    if completed_count<1 or (not allow_fewer and len(completed)<completed_count):
        raise ValueError('not enough completed flights for the requested comparison')
    count=min(completed_count,len(completed))
    ranks=np.rint(np.linspace(0,len(completed)-1,count)).astype(int)
    selected=failed+[completed[i] for i in ranks]
    if not selected:
        raise ValueError('no arena-exit or completed flights available to replay')
    return selected


def visible_scene_indices(indices, page, per_page=13):
    """Keep the uploaded two-column selector inside its existing panel."""
    page=max(0,min(page,max(0,(len(indices)-1)//per_page)))
    return indices[page*per_page:(page+1)*per_page]


def stream_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def export_replay(run_dir, completed_count=13):
    import torch  # Export only; Pygame rendering does not import Torch.
    run_dir=Path(run_dir)
    summary=json.loads((run_dir/'summary.json').read_text())
    manifest=json.loads((run_dir/'manifest.json').read_text())
    scenes=select_scenes(summary,completed_count,allow_fewer=True)
    rows=[r['scene_index'] for r in scenes]
    load=lambda name:torch.load(run_dir/name,map_location='cpu',weights_only=True)
    trace=load('trajectory.pt');schedule=load('schedule.pt');initial=load('initial_state.pt')
    arrays={name:value[:,rows].numpy() for name,value in trace.items()}
    arrays.update(targets=schedule['targets'].numpy(),points_body=schedule['points_body'][:,rows].numpy(),
                  surface_half_extents=schedule['surface_half_extents'][rows].numpy(),
                  rotor_positions=initial['rotor_positions'][rows].numpy(),inertia=initial['inertia'][rows].numpy())
    out=run_dir/'playback';out.mkdir(exist_ok=True)
    path=out/'playlist.json'
    np.savez_compressed(path.with_suffix('.npz'),**arrays)
    metadata=dict(manifest['config'],scenes=scenes,checkpoint_update=manifest['checkpoint_update'],
        model_sha256=manifest['model_sha256'],run_dir=str(run_dir.resolve()),
        selection='All arena exits; completed flights at equally spaced mass ranks, not ranked by tracking quality.',
        trajectory_sha256=stream_sha256(run_dir/'trajectory.pt'))
    path.write_text(json.dumps(metadata,indent=2,allow_nan=False)+'\n')
    return path


class Playback:
    def __init__(self, metadata):
        self.meta=metadata;self.index=0;self.seconds=0.;self.paused=False
        self.speed=1.;self.auto=True;self.hold=0.

    @property
    def scene(self): return self.meta['scenes'][self.index]
    @property
    def end_seconds(self):
        failure=self.scene['failure_step']
        return failure*self.meta['dt'] if failure is not None else self.meta['duration_seconds']
    @property
    def frame(self): return int(self.seconds/self.meta['dt']+1e-7)

    def choose(self,index):
        self.index=index%len(self.meta['scenes']);self.seconds=0.;self.hold=0.

    def seek(self,seconds):
        self.seconds=min(max(float(seconds),0.),self.end_seconds);self.hold=0.

    def advance(self,wall_seconds):
        if self.paused:return
        if self.seconds<self.end_seconds:
            self.seconds=min(self.end_seconds,self.seconds+wall_seconds*self.speed)
        elif self.auto:
            self.hold+=wall_seconds
            if self.hold>=2.:self.choose(self.index+1)


def rotation(q):
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


class Camera:
    def __init__(self):self.yaw=-.78;self.elevation=.42;self.zoom=90.

    def project(self,points,center=None,origin=(602,424),scale=None):
        points=np.asarray(points)-np.asarray([0.,0.,2.] if center is None else center)
        c,s=math.cos(self.yaw),math.sin(self.yaw)
        horizontal=c*points[...,0]-s*points[...,1]
        depth=s*points[...,0]+c*points[...,1]
        vertical=math.cos(self.elevation)*points[...,2]-math.sin(self.elevation)*depth
        factor=self.zoom if scale is None else scale
        return np.stack((origin[0]+factor*horizontal,origin[1]-factor*vertical),-1)


def run_player(path, *, screenshot=None, scene_id=None, seconds=0., max_frames=None, fps=30):
    import pygame as pg
    if not 10<=fps<=60:
        raise ValueError('replay fps must be in [10,60]')
    path=Path(path);meta=json.loads(path.read_text())
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as archive:
        data={k:archive[k] for k in ('position','orientation','velocity','omega',
                                   'force_world','points_body','rotor_positions',
                                   'surface_half_extents','targets')}
    state=Playback(meta);camera=Camera()
    if scene_id is not None:
        state.choose(next(i for i,r in enumerate(meta['scenes']) if r['scene_id']==scene_id))
    state.seek(seconds)
    pg.display.init();pg.font.init()
    screen=pg.display.set_mode((1500,950))
    duration_label=f"{meta['duration_seconds']:g}"
    pg.display.set_caption(f"DiffPhys | saved Actor #{meta['checkpoint_update']} | {duration_label}s | {len(meta['scenes'])} scenes")
    font_path=pg.font.match_font('notosanscjksc,notosanscjk,microsoftyahei,pingfangsc,droidsansfallback')
    fonts={size:pg.font.Font(font_path,size) for size in (14,16,18,21,27)}
    clock=pg.time.Clock();running=True;dragging=False;seeking=False;frames=0
    bg=(15,21,31);panel=(22,31,44);line=(56,72,90);white=(230,239,247)
    muted=(157,178,196);red=(252,115,118);green=(91,220,166);blue=(100,175,244)
    gold=(255,207,93);pink=(244,128,220)
    controls=[];scene_buttons=[]
    scene_pages=[0,0];last_scene=None
    timeline=pg.Rect(28,792,1442,18)

    def text(value,xy,size=18,color=white):
        screen.blit(fonts[size].render(str(value),True,color),xy)

    def stroke(points,color,width=1,closed=False):
        pts=np.asarray(points)
        if len(pts)>1 and np.isfinite(pts).all():pg.draw.lines(screen,color,closed,pts.astype(int).tolist(),width)

    def arrow(a,b,color,width=2):
        a=np.asarray(a);b=np.asarray(b);stroke([a,b],color,width)
        delta=b-a;n=np.linalg.norm(delta)
        if n>1:
            u=delta/n;v=np.array([-u[1],u[0]])
            pg.draw.polygon(screen,color,np.array([b,b-10*u+5*v,b-10*u-5*v]).astype(int))

    def box(center,half,project,color,rot=None):
        signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        points=signs*np.asarray(half)
        if rot is not None:points=points@rot.T
        pixels=project(points+center)
        for a in range(8):
            for bit in (1,2,4):
                b=a^bit
                if a<b:stroke([pixels[a],pixels[b]],color)

    def button(label,rect,action,active=False,color=blue):
        rect=pg.Rect(rect);controls.append((rect,action))
        pg.draw.rect(screen,(38,61,80) if active else (32,44,59),rect,border_radius=6)
        if active:pg.draw.rect(screen,color,rect,width=1,border_radius=6)
        label_surface=fonts[16].render(label,True,white)
        screen.blit(label_surface,label_surface.get_rect(center=rect.center))

    def drone(j,f,project,inset=False):
        p=data['position'][f,j];rot=rotation(data['orientation'][f,j]);arm=state.scene['arm_length_m']
        rotors=data['rotor_positions'][j]
        pixels=project(rotors@rot.T+p);com=project(p)
        if inset:box(p,data['surface_half_extents'][j],project,line,rot)
        for k,pixel in enumerate(pixels):
            color=red if k in (0,1) else blue
            stroke([com,pixel],color,3 if inset else 2)
            angles=np.linspace(0,2*math.pi,17)
            circle=rotors[k]+np.stack((np.cos(angles)*arm*.14,np.sin(angles)*arm*.14,np.zeros_like(angles)),-1)
            stroke(project(circle@rot.T+p),color,1,True)
        pg.draw.circle(screen,white,com.astype(int),4 if inset else 3)
        if inset:arrow(com,project(p+rot[:,2]*arm*.55),gold)
        at_end=state.seconds>=state.end_seconds
        force=data['force_world'][min(f,len(data['force_world'])-1),j] if not at_end else np.zeros(3)
        magnitude=np.linalg.norm(force)
        if magnitude>0:
            pulse=min(int(state.seconds/meta['force_period_seconds']),len(data['points_body'])-1)
            point=p+rot@data['points_body'][pulse,j]
            distance=arm*.9 if inset else .48
            end=point+force/magnitude*distance
            pg.draw.circle(screen,pink,project(point).astype(int),4)
            arrow(project(point),project(end),pink,3)

    def handle(action):
        if action=='pause':state.paused=not state.paused
        elif action=='previous':state.choose(state.index-1)
        elif action=='next':state.choose(state.index+1)
        elif action=='replay':state.seek(0)
        elif action=='end':state.seek(max(0,state.end_seconds-5))
        elif action=='auto':state.auto=not state.auto
        elif action=='slower':state.speed=max(.125,state.speed/2)
        elif action=='faster':state.speed=min(8,state.speed*2)
        elif action=='camera':camera.__init__()
        elif action=='capture':pg.image.save(screen,str(path.parent/'capture.png'))
        elif action.startswith('page:'):
            _,col,delta=action.split(':')
            col=int(col)
            count=sum((not r['completed']) if col==0 else r['completed'] for r in meta['scenes'])
            scene_pages[col]=max(0,min(scene_pages[col]+int(delta),max(0,(count-1)//13)))

    while running:
        elapsed=min(clock.tick(fps)/1000.,.1)
        for event in pg.event.get():
            if event.type==pg.QUIT:running=False
            elif event.type==pg.KEYDOWN:
                actions={pg.K_SPACE:'pause',pg.K_LEFT:'previous',pg.K_RIGHT:'next',pg.K_r:'replay',
                    pg.K_f:'end',pg.K_a:'auto',pg.K_MINUS:'slower',pg.K_EQUALS:'faster',
                    pg.K_c:'camera',pg.K_s:'capture'}
                if event.key in actions:handle(actions[event.key])
                elif event.key==pg.K_ESCAPE:running=False
                elif event.key==pg.K_UP:state.seek(state.seconds+5)
                elif event.key==pg.K_DOWN:state.seek(state.seconds-5)
            elif event.type==pg.MOUSEBUTTONDOWN and event.button==1:
                if timeline.inflate(0,26).collidepoint(event.pos):
                    seeking=True;state.seek((event.pos[0]-timeline.x)/timeline.width*meta['duration_seconds'])
                elif any(r.collidepoint(event.pos) for r,_ in scene_buttons):
                    state.choose(next(i for r,i in scene_buttons if r.collidepoint(event.pos)))
                elif any(r.collidepoint(event.pos) for r,_ in controls):
                    handle(next(a for r,a in controls if r.collidepoint(event.pos)))
                elif 110<event.pos[1]<760 and event.pos[0]<1000:dragging=True
            elif event.type==pg.MOUSEBUTTONUP and event.button==1:dragging=False;seeking=False
            elif event.type==pg.MOUSEMOTION:
                if seeking:state.seek((event.pos[0]-timeline.x)/timeline.width*meta['duration_seconds'])
                elif dragging:
                    camera.yaw+=event.rel[0]*.006
                    camera.elevation=float(np.clip(camera.elevation+event.rel[1]*.005,-.15,1.3))
            elif event.type==pg.MOUSEWHEEL:camera.zoom=float(np.clip(camera.zoom*1.12**event.y,45,220))
        if not seeking and not screenshot:state.advance(elapsed)
        j=state.index;f=state.frame;row=state.scene
        target_index=min(int(state.seconds/meta['target_period_seconds']),len(data['targets'])-1)
        target=data['targets'][target_index]
        p=data['position'][f,j];v=data['velocity'][f,j];w=data['omega'][f,j]
        end=state.seconds>=state.end_seconds
        color=green if row['completed'] else red
        screen.fill(bg);controls=[];scene_buttons=[]
        pg.draw.rect(screen,panel,(1000,0,500,774))
        text('长时飞行回放 · 换目标与表面脉冲', (24,16),27)
        force_label=(f"{100*meta['force_fraction']:g}% mg × {meta['pulse_seconds']:g}s / {meta['force_period_seconds']:g}s"
                     if 'force_fraction' in meta and 'pulse_seconds' in meta else '力参数以回放记录为准')
        text(f"固定 Actor #{meta['checkpoint_update']}  |  已保存轨迹  |  {force_label}",(25,57),16,muted)
        text(f"Scene {row['scene_id']}   {'完成 '+duration_label+' 秒' if row['completed'] else '越界案例'}   {state.seconds:05.2f} / {state.end_seconds:.2f} s",(25,87),21,color)
        screen.set_clip(pg.Rect(8,120,984,642))
        side=meta['arena_side'];center=np.array([0.,0.,side/2]);half=side/2
        project=lambda points:camera.project(points,center=center)
        for value in np.linspace(-half,half,9):
            stroke(project([[value,-half,0],[value,half,0]]),(34,46,61))
            stroke(project([[-half,value,0],[half,value,0]]),(34,46,61))
        box(center,[half]*3,project,(183,119,76))
        box(center,[half-meta['target_margin']]*3,project,(55,85,119))
        trail=data['position'][:f+1:max(1,math.ceil((f+1)/800)),j]
        stroke(project(trail),(66,135,140),2)
        if f:stroke(project([trail[-1],p]),(66,135,140),2)
        for axis,axis_color in enumerate((red,green,blue)):
            base=np.array([-half,-half,0.]);tip=base.copy();tip[axis]+=.55
            arrow(project(base),project(tip),axis_color)
        goal=project(target);pg.draw.circle(screen,gold,goal.astype(int),9,2)
        for delta in ([-13,0],[0,-13]):stroke([goal+delta,goal-np.array(delta)],gold)
        stroke(project([p,target]),(93,84,57))
        drone(j,f,project)
        text('目标 '+str(target_index+1),goal+[12,-15],16,gold)
        if end and not row['completed']:
            pg.draw.circle(screen,red,project(p).astype(int),14,3)
        screen.set_clip(None)
        # Close-up shows the same attitude and point force, with a displayed scale.
        inset=pg.Rect(23,489,305,246)
        pg.draw.rect(screen,panel,inset,border_radius=8)
        pg.draw.rect(screen,line,inset,1,border_radius=8)
        text('无人机近景 · 姿态 / 作用点', (36,501),16)
        scale=95/max(float(np.linalg.norm(data['rotor_positions'][j],axis=-1).max()),.001)
        screen.set_clip(inset.inflate(-4,-4))
        close=lambda points:camera.project(points,center=p,origin=(176,628),scale=scale)
        drone(j,f,close,True)
        screen.set_clip(None)
        text(f'近景放大 {scale/camera.zoom:.1f}× · 紫箭头仅表示方向',(34,708),14,muted)
        text(f"橙：{side:g}m 场地   蓝：{side-2*meta['target_margin']:g}m 目标区   金：目标",(365,719),16,muted)
        text('拖动旋转视角 · 滚轮缩放 · C 复位',(365,744),16,muted)
        text('点击编号切换无人机', (1020,20),27)
        failed=[i for i,r in enumerate(meta['scenes']) if not r['completed']]
        completed=[i for i,r in enumerate(meta['scenes']) if r['completed']]
        text(f'越界 {len(failed)} 架', (1020,70),18,red)
        text(f'完成 {len(completed)} 架', (1252,70),18,green)
        if last_scene!=j:
            for col,indices in enumerate((failed,completed)):
                if j in indices:scene_pages[col]=indices.index(j)//13
            last_scene=j
        for col,indices in enumerate((failed,completed)):
            x=1016+col*235
            button('<',(x+145,70,32,26),f'page:{col}:-1')
            button('>',(x+184,70,32,26),f'page:{col}:1')
            for rank,i in enumerate(visible_scene_indices(indices,scene_pages[col])):
                r=meta['scenes'][i];rect=pg.Rect(1016+col*235,104+rank*32,223,28)
                scene_buttons.append((rect,i))
                pg.draw.rect(screen,(51,67,83) if i==j else (29,40,54),rect,border_radius=4)
                if i==j:pg.draw.rect(screen,red if col==0 else green,rect,2,border_radius=4)
                mass=f"{r['mass_kg']*1000:.0f}g" if r['mass_kg']<1 else f"{r['mass_kg']:.2f}kg"
                finish=r['failure_step']*meta['dt'] if r['failure_step'] is not None else meta['duration_seconds']
                text(f"{r['scene_id']:3}   {mass:>6}   {finish:.2f}s",(rect.x+9,rect.y+2),16,red if col==0 else green)
        text(f"质量 {row['mass_kg']*1000:.2f} g    臂长 {row['arm_length_m']*100:.2f} cm",(1019,534),18)
        text(f"推重比 {row['thrust_to_weight']:.2f}    T/I {row['torque_to_inertia']:.1f}",(1019,564),16,muted)
        text(f"|p−目标|  {np.linalg.norm(p-target)*100:.2f} cm",(1019,599),21,gold)
        text(f"|v|  {np.linalg.norm(v):.4f} m/s     |ω|  {np.linalg.norm(w):.4f} rad/s",(1019,633),18)
        text(f"位置  [{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}] m",(1019,665),16,muted)
        step=min(f,len(data['force_world'])-1)
        force=0. if end else float(np.linalg.norm(data['force_world'][step,j]))
        text(f"{'脉冲 ON' if force else '脉冲 OFF'}   F={force:.4f} N",(1019,695),18,pink if force else muted)
        status=('质心越界 · 停在首个越界帧' if not row['completed'] else '完整到达 '+duration_label+' 秒') if end else '飞行中 · 记忆连续'
        text(status,(1019,733),18,color)
        # The full 60 s axis remains visible; post-failure padding is darkened.
        pg.draw.rect(screen,(49,64,79),timeline,border_radius=5)
        valid_width=round(timeline.width*state.end_seconds/meta['duration_seconds'])
        pg.draw.rect(screen,(69,100,116),(timeline.x,timeline.y,valid_width,timeline.height),border_radius=5)
        x=timeline.x+round(timeline.width*state.seconds/meta['duration_seconds'])
        pg.draw.circle(screen,gold,(x,timeline.centery),9)
        for second in np.linspace(0,meta['duration_seconds'],7):
            x=timeline.x+round(timeline.width*second/meta['duration_seconds'])
            pg.draw.line(screen,muted,(x,784),(x,812));text(f'{second:.2g}s',(x-10,763),14,muted)
        if not row['completed']:
            x=timeline.x+valid_width;pg.draw.line(screen,red,(x,782),(x,814),3)
        items=[('上一架','previous',110),('下一架','next',110),('播放' if state.paused else '暂停','pause',90),
            ('重播','replay',80),('越界前5秒' if not row['completed'] else '最后5秒','end',130),
            ('自动轮播 '+('开' if state.auto else '关'),'auto',135),('减速','slower',80),
            ('加速','faster',80),('视角复位','camera',110),('截图','capture',80)]
        x=27
        for label,action,width in items:
            button(label,(x,832,width,36),action,(action=='auto' and state.auto));x+=width+9
        text(f'{state.speed:g}×',(x+7,838),18,gold)
        text('← / → 切换   空格 暂停   ↑ / ↓ 前后5秒   F 越界前5秒   拖动时间轴跳转   A 自动轮播',(27,886),18,muted)
        text('紫色点为真实表面作用点；箭头长度不代表力大小。越界后不播放冻结填充。完成全程不等于每时刻都稳态悬停。',(27,916),14,muted)
        pg.display.flip();frames+=1
        if screenshot:
            pg.image.save(screen,str(screenshot));running=False
        if max_frames and frames>=max_frames:running=False
    pg.quit()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-dir',type=Path,help='export saved EVAL then play')
    group.add_argument('--replay',type=Path,help='previously exported playlist.json; NumPy/Pygame only')
    parser.add_argument('--completed-count',type=int,default=13)
    parser.add_argument('--export-only',action='store_true')
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--scene-id',type=int)
    parser.add_argument('--seconds',type=float,default=0.)
    parser.add_argument('--max-frames',type=int,help='bounded UI smoke run')
    parser.add_argument('--fps',type=int,default=30,help='software replay render cap, 10..60')
    args=parser.parse_args()
    path=export_replay(args.run_dir,args.completed_count) if args.run_dir else args.replay
    print('Replay:',path,flush=True)
    if not args.export_only:run_player(path,screenshot=args.screenshot,scene_id=args.scene_id,
                                     seconds=args.seconds,max_frames=args.max_frames,fps=args.fps)


if __name__=='__main__':main()
