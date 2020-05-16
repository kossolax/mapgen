from MapGenerator import MapGenerator
from VMFWriter import VMFWriter
import PyQt5

gen = MapGenerator(200, 200)
gen.generate()


vmf = VMFWriter("out.vmf", 48, 3)
x_offset = int(gen.rows / 2)
y_offset = int(gen.cols / 2)

for i in gen.optimize():
    vmf.add_cuboid(
        pointsMin=(i.max[0] - x_offset, (i.min[1] - y_offset)*-1, 0),
        pointsMax=(i.min[0] - x_offset, (i.max[1] - y_offset)*-1, 1)
    )


vmf.save()
