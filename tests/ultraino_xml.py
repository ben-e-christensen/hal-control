"""Minimal java.beans.XMLDecoder interpreter for Ultraino scene files.

Supports the subset Ultraino uses: constructors, properties (get/set),
Class.getField(name) + Field.set/get, Map.put, List.add, id/idref with
lazy (forward) resolution. Objects become dicts. Only used by tests to read
transducer positions and orderNumber out of emptyboard.xml.gz.
"""

import gzip
import xml.etree.ElementTree as ET

_PRIM = {"float": float, "double": float, "int": int, "long": int, "short": int,
         "string": lambda s: s or "", "boolean": lambda s: s == "true"}


class Obj(dict):
    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    __hash__ = object.__hash__

    def __eq__(self, other):
        return self is other


class _Field:
    def __init__(self, name):
        self.name = name


def decode(path):
    """Return {id: object} for every element with an id attribute."""
    ids, pending = {}, {}

    def lookup(key):
        if key not in ids:
            pending[key]()
        return ids[key]

    def run(el, owner):
        if el.tag in _PRIM:
            return _PRIM[el.tag](el.text)
        if el.tag == "null":
            return None
        if el.tag == "array":
            items = []
            for c in el:
                run(c, lambda: items)
            return items
        a = el.attrib
        if "idref" in a and len(el) == 0:
            return lookup(a["idref"])
        args, state = [], {}

        def value():
            if "v" not in state:
                state["v"] = invoke(a, args, owner)
                if "id" in a:
                    ids[a["id"]] = state["v"]
            return state["v"]

        if "id" in a:
            pending[a["id"]] = value
        for c in el:
            if c.tag == "void":
                run(c, value)       # statement on this element's value
            else:
                args.append(run(c, value))
        return value()

    def invoke(a, args, owner):
        if "idref" in a:
            return lookup(a["idref"])
        if "class" in a and "method" not in a and "property" not in a:
            return Obj(a["class"])
        if a.get("method") == "getField":
            return _Field(args[0])
        if "property" in a:
            tgt, prop = owner(), a["property"]
            if args:
                tgt[prop] = args[0]
                return None
            if tgt is None:
                return Obj(prop)
            return tgt.setdefault(prop, Obj(prop))
        m, tgt = a.get("method"), owner()
        if isinstance(tgt, _Field):
            if m == "set":
                args[0][tgt.name] = args[1]
                return None
            if m == "get":
                return args[0].setdefault(tgt.name, Obj(tgt.name))
        if m == "put":
            tgt[args[0]] = args[1]
            return None
        if m == "add":
            tgt.setdefault("_items", []).append(args[-1])
        return None

    root = ET.parse(gzip.open(path)).getroot()
    for c in root:
        run(c, lambda: None)
    return ids


def transducers(path):
    """List of dicts (name, order, pos) for every Transducer, sorted by orderNumber.

    Java beans omit default values, so a missing orderNumber or coordinate is 0.
    """
    out = []
    for v in decode(path).values():
        if isinstance(v, Obj) and v.cls.endswith(".Transducer"):
            t = v.get("transform", {}).get("translation", {})
            out.append({"name": v.get("name"), "order": v.get("orderNumber", 0),
                        "pos": (t.get("x", 0.0), t.get("y", 0.0), t.get("z", 0.0))})
    return sorted(out, key=lambda t: t["order"])
