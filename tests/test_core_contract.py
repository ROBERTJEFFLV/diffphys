"""Unchanged mathematical kernels checked against the pre-cleanup source AST."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


class Canonical(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0],ast.Expr)
                and isinstance(node.body[0].value,ast.Constant)
                and isinstance(node.body[0].value.value,str)):
            node.body=node.body[1:]
        return node
    def visit_Name(self,node):
        if node.id.startswith('L2F'):node.id=node.id.replace('L2F','Raptor')
        return node


def canonical_digest(node):
    # ast.dump omits empty fields differently in Python 3.13. Serialize explicit
    # fields instead; empty type_params is new metadata, not a kernel change.
    def encode(value):
        if isinstance(value,ast.AST):
            assert not getattr(value,'type_params',[])
            return [type(value).__name__, [[k,encode(v)] for k,v in ast.iter_fields(value)
                                          if k != 'type_params']]
        if isinstance(value,list):
            return [encode(x) for x in value]
        if value is Ellipsis:
            return ['Ellipsis']
        return value
    payload=json.dumps(encode(Canonical().visit(node)),separators=(',',':'))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_existing_physics_actor_loss_time_decay_and_adam_kernels_are_preserved():
    contract=json.loads((ROOT/'tests/core_contract.json').read_text())
    for name,symbols in contract['symbols'].items():
        for qualified,expected in symbols.items():
            node=ast.parse((ROOT/name).read_text())
            for symbol in qualified.split('.'):
                node=next(x for x in node.body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==symbol)
            actual=canonical_digest(node)
            assert actual==expected,(name,qualified)
