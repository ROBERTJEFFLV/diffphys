from datetime import datetime, timezone
from pathlib import Path
import json, os, signal, subprocess, sys
work = Path(sys.argv[1]).resolve()
info = json.loads((work/'active_launch.json').read_text())
def write(name, data):
    temporary = work/(name+'.tmp')
    temporary.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(temporary, work/name)
env = {**os.environ, **info['environment']}
with (work/'training.stdout.log').open('a', buffering=1) as output:
    trainer = subprocess.Popen(info['command'], cwd=info['root'], env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
    write('processes.json', {'training_supervisor':os.getpid(), 'trainer':trainer.pid})
    def stop(signum, frame):
        if trainer.poll() is None:
            trainer.send_signal(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    result = trainer.wait()
summary_path = work/'summary.json'
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
write('training_exit.json', {'exit_code':result, 'finished_at_utc':datetime.now(timezone.utc).isoformat(), 'summary':summary})
