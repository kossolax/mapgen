

class VMFWriter:
    def __init__(self, file, grid, scale):
        self.file = file
        self.grid = grid
        self.scale = scale*grid
        self.listOfBlocks = []

    def add_cuboid(self, pointsMin, pointsMax, texture=None):
        pointsMin = (pointsMin[0] * self.scale, pointsMin[1] * self.scale, pointsMin[2] * self.scale)
        pointsMax = (pointsMax[0] * self.scale, pointsMax[1] * self.scale, pointsMax[2] * self.scale)

        top = Plane(
            (pointsMin[0], pointsMin[1], pointsMax[2]),
            (pointsMin[0], pointsMax[1], pointsMax[2]),
            (pointsMax[0], pointsMax[1], pointsMax[2]),
            ((1, 0, 0), (0, -1, 0)),
            "ECO/GRASS"
        )
        bot = Plane(
            (pointsMin[0], pointsMax[1], pointsMin[2]),
            (pointsMin[0], pointsMin[1], pointsMin[2]),
            (pointsMax[0], pointsMin[1], pointsMin[2]),
            ((1, 0, 0), (0, -1, 0))
        )
        left = Plane(
            (pointsMin[0], pointsMin[1], pointsMin[2]),
            (pointsMin[0], pointsMax[1], pointsMin[2]),
            (pointsMin[0], pointsMax[1], pointsMax[2]),
            ((0, 1, 0), (0, 0, -1)),
            "ECO/GRASSEDGE"
        )
        right = Plane(
            (pointsMax[0], pointsMax[1], pointsMin[2]),
            (pointsMax[0], pointsMin[1], pointsMin[2]),
            (pointsMax[0], pointsMin[1], pointsMax[2]),
            ((0, 1, 0), (0, 0, -1)),
            "ECO/GRASSEDGE"
        )
        front = Plane(
            (pointsMin[0], pointsMax[1], pointsMin[2]),
            (pointsMax[0], pointsMax[1], pointsMin[2]),
            (pointsMax[0], pointsMax[1], pointsMax[2]),
            ((1, 0, 0), (0, 0, -1)),
            "ECO/GRASSEDGE"
        )
        back = Plane(
            (pointsMax[0], pointsMin[1], pointsMin[2]),
            (pointsMin[0], pointsMin[1], pointsMin[2]),
            (pointsMin[0], pointsMin[1], pointsMax[2]),
            ((1, 0, 0), (0, 0, -1)),
            "ECO/GRASSEDGE"
        )

        solid = Solid([top, bot, left, right, front, back])
        self.listOfBlocks.append(solid)

    def save(self):
        stdout = open(self.file, "w")
        stdout.write('versioninfo\n')
        stdout.write('{\n')
        stdout.write('  "editorversion" "400"\n')
        stdout.write('  "editorbuild" "8456"\n')
        stdout.write('  "mapversion" "1"\n')
        stdout.write('  "formatversion" "100"\n')
        stdout.write('  "prefab" "0"\n')
        stdout.write('}\n\n')

        stdout.write('visgroups\n')
        stdout.write('{\n')
        stdout.write('}\n\n')

        stdout.write('viewsettings\n')
        stdout.write('{\n')
        stdout.write('  "bSnapToGrid" "1"\n')
        stdout.write('  "bShowGrid" "1"\n')
        stdout.write('  "bShowLogicalGrid" "0"\n')
        stdout.write('  "nGridSpacing" "{0}"\n'.format(self.grid))
        stdout.write('  "bShow3DGrid" "0"\n')
        stdout.write('}\n\n')

        stdout.write('world\n')
        stdout.write('{\n')
        stdout.write('  "id" "1"\n')
        stdout.write('  "mapversion" "1"\n')
        stdout.write('  "classname" "worldspawn"\n')
        stdout.write('  "skyname" "sky_dust"\n')
        stdout.write('  "maxpropscreenwidth" "-1"\n')
        stdout.write('  "detailvbsp" "detail.vbsp"\n')
        stdout.write('  "detailmaterial" "detail/detailsprites"\n')

        for i in self.listOfBlocks:
            stdout.write(i.dump())

        stdout.write('}\n\n')

        stdout.write('cameras\n')
        stdout.write('{\n')
        stdout.write('  "activecamera" "-1"\n')
        stdout.write('}\n\n')

        stdout.write('cordons\n')
        stdout.write('{\n')
        stdout.write('  "active" "0"\n')
        stdout.write('}\n')


class Plane:
    id = 0

    def __init__(self, A, B, C, UV, material="TOOLS/TOOLSNODRAW"):
        Plane.id += 1
        self.id = Plane.id
        self.material = material
        self.A = A
        self.B = B
        self.C = C
        self.UV = UV

    def dump(self):
        output = 'side\n'
        output += '{\n'
        output += ' "id" "{0}"\n'.format(self.id)
        output += ' "plane" "({0} {1} {2}) ({3} {4} {5}) ({6} {7} {8})"\n'.format(
            self.A[0], self.A[1], self.A[2],
            self.B[0], self.B[1], self.B[2],
            self.C[0], self.C[1], self.C[2]
        )
        output += ' "material" "{}"\n'.format(self.material)
        output += ' "uaxis" "[{0} {1} {2} 0] 0.25"\n'.format(self.UV[0][0], self.UV[0][1], self.UV[0][2])
        output += ' "vaxis" "[{0} {1} {2} 0] 0.25"\n'.format(self.UV[1][0], self.UV[1][1], self.UV[1][2])
        output += ' "rotation" "0"\n'
        output += ' "lightmapscale" "16"\n'
        output += ' "smoothing_groups" "0"\n'
        output += '}\n'
        return output


class Solid:
    id = 0

    def __init__(self, planes=[]):
        Solid.id += 1
        self.id = Solid.id
        self.planes = planes

    def add_plane(self, plane: Plane):
        self.planes.append(plane)

    def dump(self):
        output = 'solid\n'
        output += '{\n'
        output += ' "id" "{0}"\n'.format(self.id)

        for i in self.planes:
            output += i.dump()

        output += '}\n'
        return output
