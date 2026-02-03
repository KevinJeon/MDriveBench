import sys, os, json
import xml.etree.ElementTree as ET

TX = -507.434
TY =  210.084

def shift_xy_attrs_xml(inp, outp):
    tree = ET.parse(inp)
    root = tree.getroot()
    touched = 0
    for el in root.iter():
        if 'x' in el.attrib and 'y' in el.attrib:
            el.attrib['x'] = str(float(el.attrib['x']) + TX)
            el.attrib['y'] = str(float(el.attrib['y']) + TY)
            touched += 1
    tree.write(outp, encoding="utf-8", xml_declaration=True)
    print(f"[OK] XML: shifted {touched} nodes, wrote {outp}")

def shift_xy_lists_json(inp, outp):
    obj = json.load(open(inp))

    def is_num(x):
        return isinstance(x, (int, float))

    def shift_point(p):
        # point is [x,y] (possibly longer)
        if isinstance(p, list) and len(p) >= 2 and is_num(p[0]) and is_num(p[1]):
            p[0] = float(p[0]) + TX
            p[1] = float(p[1]) + TY
        return p

    touched = 0

    pts = obj.get("points")
    if isinstance(pts, list):
        for i in range(len(pts)):
            before = pts[i]
            pts[i] = shift_point(pts[i])
            if pts[i] is not before or (isinstance(before, list) and len(before) >= 2):
                touched += 1

    pls = obj.get("polylines")
    if isinstance(pls, list):
        for pl in pls:
            if isinstance(pl, list):
                for i in range(len(pl)):
                    pl[i] = shift_point(pl[i])

    obj["applied_translate"] = {"tx": TX, "ty": TY}
    obj["type"] = str(obj.get("type", "pkl-geometry")) + "-translated"

    json.dump(obj, open(outp, "w"))
    print(f"[OK] JSON: wrote {outp} (tx={TX}, ty={TY})")

def main():
    if len(sys.argv) != 3:
        print("usage: python translate.py input.(xml|json) output.(xml|json)")
        sys.exit(1)

    inp, outp = sys.argv[1], sys.argv[2]
    ext = os.path.splitext(inp)[1].lower()

    if ext == ".xml":
        shift_xy_attrs_xml(inp, outp)
    elif ext == ".json":
        shift_xy_lists_json(inp, outp)
    else:
        raise SystemExit(f"Unsupported input type: {ext} (expected .xml or .json)")

if __name__ == "__main__":
    main()
