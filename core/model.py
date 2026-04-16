import numpy as np


class OBJModel:
    def __init__(self, package):
        self.parts = dict()
        self.path = f"src/{package}"
        self.load(f"{self.path}/model.obj")

    def load(self, path):
        verts, uvs = [], []
        current_mtl = "default"

        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('vt '):
                    parts = line.strip().split()
                    uvs.append([float(parts[1]), float(parts[2])])
                elif line.startswith('usemtl '):
                    current_mtl = line.strip().split()[1]
                elif line.startswith('f '):
                    if current_mtl not in self.parts:
                        self.parts[current_mtl] = {'v': [], 'uv': []}

                    parts = line.strip().split()[1:]
                    face_v, face_uv = [], []

                    for p in parts:
                        vals = p.split('/')
                        face_v.append(int(vals[0]) - 1)
                        if len(vals) > 1 and vals[1]:
                            face_uv.append(int(vals[1]) - 1)
                        else:
                            face_uv.append(0)

                    idx = [0, 1, 2, 2, 3, 0] if len(face_v) == 4 else [0, 1, 2]
                    for i in idx:
                        self.parts[current_mtl]['v'].append(verts[face_v[i]])
                        self.parts[current_mtl]['uv'].append(uvs[face_uv[i]] if uvs else [.0, .0])

        for mtl in self.parts:
            self.parts[mtl]['v'] = np.array(self.parts[mtl]['v'], dtype=np.float32)
            self.parts[mtl]['uv'] = np.array(self.parts[mtl]['uv'], dtype=np.float32)


if __name__ == '__main__':
    model = OBJModel(package="mercury")
    parts, *_ = model.parts.values()
    print(np.max(parts['v']) - np.min(parts['v']))
