import numpy as np
import cv2
import noise
from opensimplex import OpenSimplex
import random
import math
from enum import Enum

parameters = {
    "shape": {
        "scale": 8,
        "round": 0.5,
        "inflate": 0.4,
        "octave": 4
    },
    "numRivers": 10,
    "drainageSeed": 0,
    "riverSeed": 0,
    "biomeBias": {
        "north_temperature": 0.5,
        "south_temperature": 0.5,
        "moisture": 0
    }
}


def mix(a, b, t):
    return a * (1.0 - t) + b * t


class Biomes(Enum):
    OCEAN = (6, 66, 115)

    MARSH = (118, 182, 196)
    ICE = (222, 243, 246)
    LAKE = (127, 205, 255)

    BEACH = (236, 209, 168)

    SNOW = (252, 252, 252)
    TUNDRA = (192, 230, 237)
    BARE = (222, 247, 242)
    SCORCHED = (243, 252, 255)

    TAIGA = (58, 156, 27)
    SHRUBLAND = (205, 214, 129)
    TEMPERATE_DESERT = (154, 148, 116)

    TEMPERATE_RAIN_FOREST = (35, 77, 32)
    TEMPERATE_DECIDUOUS_FOREST = (0, 127, 14)
    GRASSLAND = (119, 171, 89)
    #    TEMPERATE_DESERT = (201, 223, 138)

    TROPICAL_RAIN_FOREST = (73, 103, 75)
    TROPICAL_SEASONAL_FOREST = (148, 130, 58)
    #    GRASSLAND = (119, 171, 89)
    SUBTROPICAL_DESERT = (229, 205, 158)


class Node:
    def __init__(self, map, x: int, y: int, value: int = 0):
        self.map = map
        self.x = x
        self.y = y
        self.value = value
        self.is_ocean = False
        self.is_coast = False
        self.is_lake = False
        self.is_water = False
        self.is_spring = False
        self.slope = None
        self.river_flow = 0
        self.distanceToCoast = 9999999999
        self.distanceToWater = 9999999999
        self.moisture = 0
        self.temperature = 0
        self.elevation = 0
        self.biome = None

    def __str__(self):
        return "{0}x{1}".format(self.x, self.y)

    def __int__(self):
        def water():
            if self.is_ocean:
                return 100 - int(-self.distanceToCoast * 100)
            if self.is_lake:
                return 150
            if self.is_water:
                return 200
            if self.is_coast:
                return 255
            return int(self.distanceToCoast * 255)

        def elevation():
            if self.is_coast:
                return 255
            if self.is_water:
                return min(255, int(self.river_flow * 10))
            return int(self.elevation * 255)

        def moisture():
            if self.is_water:
                return 255
            return int(self.moisture * 255)

        def temperature():
            return int(self.temperature * 255)

        if self.map.debug == "water":
            return water()
        if self.map.debug == "elevation":
            return elevation()
        if self.map.debug == "moisture":
            return moisture()
        if self.map.debug == "temperature":
            return temperature()
        return 0


class Mesh:
    def __init__(self):
        self.nodes = []
        self.min = (0, 0)
        self.max = (0, 0)

    def add(self, node):
        if len(self.nodes) == 0:
            self.min = node.x, node.y
            self.max = node.x+1, node.y+1
        else:
            self.min = min(self.min[0], node.x), min(self.min[1], node.y)
            self.max = max(self.max[0], node.x), max(self.max[1], node.y)

        self.nodes.append(node)


class MapGenerator:
    def __init__(self, rows, cols):
        self.debug = None
        self.map = np.empty((rows, cols), dtype=Node)
        for x in range(0, rows):
            for y in range(0, cols):
                self.map[x][y] = Node(self, x, y, 128)

        self.rows = rows
        self.cols = cols

    def generate(self):
        def _water(size=2):
            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    if x < size or x >= self.rows - size or y < size or y >= self.cols - size:
                        self.map[x][y].is_water = True

                    dx = (x - self.rows / 2) / (self.rows / 2)
                    dy = (y - self.cols / 2) / (self.cols / 2)
                    d = max(abs(dx), abs(dy)) ** 2
                    n = noise.pnoise2(x / self.rows * parameters["shape"]["scale"],
                                      y / self.cols * parameters["shape"]["scale"],
                                      octaves=parameters["shape"]["octave"])
                    n = mix(n, 0.5, parameters["shape"]["round"])
                    w = n - (1.0 - parameters["shape"]["inflate"]) * d

                    if w < 0:
                        self.map[x][y].is_water = True

            self.map[0][0].is_water = True

        def _erode(size=2):

            visited = dict()
            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    if self.map[x][y].is_water is True:
                        visited[self.map[x][y]] = True
                    elif self.map[x][y] not in visited:
                        length = 1
                        stack = []
                        current = dict()

                        current[self.map[x][y]] = True
                        visited[self.map[x][y]] = True
                        stack.append(self.map[x][y])

                        while len(stack) > 0:
                            n = stack.pop(0)
                            for _x, _y in self.edge(n.x, n.y):
                                if self.map[_x][_y].is_water is True:
                                    visited[self.map[_x][_y]] = True
                                elif self.map[_x][_y] not in visited:
                                    current[self.map[_x][_y]] = True
                                    visited[self.map[_x][_y]] = True
                                    stack.append(self.map[_x][_y])
                                    length += 1

                        if length <= size:
                            for n in current.keys():
                                self.map[n.x][n.y].is_water = True

        def _ocean():
            stack = [(0, 0)]
            while len(stack) > 0:
                _x, _y = stack.pop(0)
                for x, y in self.edge(_x, _y):
                    if self.map[x][y].is_ocean is False and self.map[x][y].is_water is True:
                        self.map[x][y].is_ocean = True
                        stack.append((x, y))

            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    if self.map[x][y].is_ocean:
                        for _x, _y in self.edge(x, y):
                            if self.map[_x][_y].is_water is False:
                                self.map[_x][_y].is_coast = True
                    elif self.map[x][y].is_water:
                        self.map[x][y].is_lake = True

        def _elevation():
            stack = []
            distance_min = 1
            distance_max = 1

            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    if self.map[x][y].is_coast:
                        self.map[x][y].distanceToCoast = 0
                        stack.append(self.map[x][y])

            while len(stack) > 0:
                n = stack.pop(0)

                for _x, _y in self.edge(n.x, n.y):
                    p = self.map[_x][_y]
                    r = mix(1, abs(noise.pnoise2(_x / self.rows * parameters["shape"]["scale"] * 2,
                                                 _y / self.cols * parameters["shape"]["scale"] * 2, octaves=4)), 0.9)
                    d = (0 if p.is_lake else r) + n.distanceToCoast

                    if d < p.distanceToCoast:
                        p.distanceToCoast = d
                        p.slope = n

                        if p.is_ocean is False and d > distance_max:
                            distance_max = d
                        if p.is_ocean is True and d > distance_min:
                            distance_min = d

                        if p.is_lake:
                            stack.insert(0, p)
                        else:
                            stack.append(p)

            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    p = self.map[x][y]
                    p.distanceToCoast = (-p.distanceToCoast / distance_min) if p.is_ocean else (
                            p.distanceToCoast / distance_max)

        # TODO: Fix lakes
        def _normalize_elevation(scale=1.1):
            nonocean = []
            elevation = {}

            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    if self.map[x][y].is_ocean is False:
                        nonocean.append(self.map[x][y])

            nonocean = sorted(nonocean, key=lambda i: i.distanceToCoast)
            for i, n in enumerate(nonocean):
                y = i / (len(nonocean) - 1)
                x = math.sqrt(scale) - math.sqrt(scale * (1 - y))
                elevation[n] = x if x < 1 else 1

            for i, n in enumerate(nonocean):
                t = 0
                cpt = 0
                for x, y in self.edge(n.x, n.y):
                    t += (elevation[self.map[x][y]] if self.map[x][y] in elevation else 0)
                    cpt += 1

                n.elevation = t / cpt

        def _river(min_spring, max_spring, river=30):
            springs = []

            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    if min_spring <= self.map[x][y].elevation <= max_spring:
                        springs.append(self.map[x][y])

            random.shuffle(springs)
            springs = springs[0:river]

            for n in springs:
                n.is_spring = True
                cpt = 0
                while n is not None:
                    cpt += 1
                    n.river_flow = cpt
                    n.is_water = True
                    n = n.slope

        def _moisture():
            stack = []
            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    if self.map[x][y].is_water is True:
                        self.map[x][y].distanceToWater = 0
                        stack.append(self.map[x][y])

            distance_max = 1
            while len(stack) > 0:
                n = stack.pop(0)

                for _x, _y in self.edge(n.x, n.y):
                    p = self.map[_x][_y]
                    if p.is_water is False:
                        d = n.distanceToWater + 1

                        if d < p.distanceToWater:
                            p.distanceToWater = d
                            if d > distance_max:
                                distance_max = d
                            stack.append(p)

            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    p = self.map[x][y]
                    p.moisture = 1 if p.is_water else 1 - math.sqrt(p.distanceToWater / distance_max)

        def _normalize_moisture():
            lands = []
            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    if self.map[x][y].is_water is False:
                        lands.append(self.map[x][y])

            lands = sorted(lands, key=lambda i: i.moisture)
            for i, n in enumerate(lands):
                n.moisture = parameters["biomeBias"]["moisture"] + 1 * i / (len(lands) - 1)

        def _temperature():
            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    l = x / (self.cols - 1)
                    t = mix(parameters["biomeBias"]["north_temperature"], parameters["biomeBias"]["south_temperature"],
                            l)
                    self.map[x][y].temperature = min(1, max(0, 1 - self.map[x][y].elevation + t))

        def _biome():
            for x in range(0, self.rows):
                for y in range(0, self.cols):
                    n = self.map[x][y]

                    if n.is_ocean is True:
                        n.biome = Biomes.OCEAN
                    elif n.is_water is True:
                        if n.temperature > 0.9:
                            n.biome = Biomes.MARSH
                        elif n.temperature < 0.2:
                            n.biome = Biomes.ICE
                        else:
                            n.biome = Biomes.LAKE
                    elif n.is_coast is True:
                        n.biome = Biomes.BEACH
                    elif n.temperature < 0.2:
                        if n.moisture > 0.5:
                            n.biome = Biomes.SNOW
                        elif n.moisture > 0.33:
                            n.biome = Biomes.TUNDRA
                        elif n.moisture > 0.16:
                            n.biome = Biomes.BARE
                        else:
                            n.biome = Biomes.SCORCHED
                    elif n.temperature < 0.4:
                        if n.moisture > 0.66:
                            n.biome = Biomes.TAIGA
                        elif n.moisture > 0.33:
                            n.biome = Biomes.SHRUBLAND
                        else:
                            n.biome = Biomes.TEMPERATE_DESERT
                    elif n.temperature < 0.7:
                        if n.moisture > 0.83:
                            n.biome = Biomes.TEMPERATE_RAIN_FOREST
                        elif n.moisture > 0.5:
                            n.biome = Biomes.TEMPERATE_DECIDUOUS_FOREST
                        elif n.moisture > 0.16:
                            n.biome = Biomes.GRASSLAND
                        else:
                            n.biome = Biomes.TEMPERATE_DESERT
                    else:
                        if n.moisture > 0.66:
                            n.biome = Biomes.TROPICAL_RAIN_FOREST
                        elif n.moisture > 0.33:
                            n.biome = Biomes.TROPICAL_SEASONAL_FOREST
                        elif n.moisture > 0.16:
                            n.biome = Biomes.GRASSLAND
                        else:
                            n.biome = Biomes.SUBTROPICAL_DESERT

        _water(2)
        _erode(8)
        _ocean()

        _elevation()
        _normalize_elevation(1.1)

        _river(0.3, 0.9, parameters["numRivers"])

        _moisture()
        _normalize_moisture()

        _temperature()
        _biome()

    def optimize(self):
        optimized = {}
        meshes = []

        for x in range(0, self.rows):
            for y in range(0, self.cols):
                if self.map[x][y] not in optimized and self.map[x][y].is_water is False:
                    m = Mesh()

                    for _x in range(x, self.rows):
                        if self.map[_x][y].biome != self.map[x][y].biome:
                            break
                        if self.map[_x][y] in optimized:
                            break

                        m.add(self.map[_x][y])
                        optimized[self.map[_x][y]] = True

                    meshes.append(m)

        print(len(meshes), len(optimized))
        return meshes


    def is_valid(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols

    def edge(self, x, y):
        edges = []
        if self.is_valid(x - 1, y):
            edges.append((x - 1, y))
        if self.is_valid(x + 1, y):
            edges.append((x + 1, y))
        if self.is_valid(x, y - 1):
            edges.append((x, y - 1))
        if self.is_valid(x, y + 1):
            edges.append((x, y + 1))
        return edges

    def show_layer(self, debug="elevation"):
        self.debug = debug
        t = self.map.astype(np.uint8)
        cv2.imwrite("Generator/map_" + self.debug + ".jpg", t)
        cv2.imshow('image' + self.debug, t)

    def show(self):
        img = np.zeros((self.rows, self.cols, 3), np.uint8)

        for x in range(0, self.rows):
            for y in range(0, self.cols):
                n = self.map[x][y].biome
                img[x][y][0] = n.value[2]
                img[x][y][1] = n.value[1]
                img[x][y][2] = n.value[0]

        cv2.imwrite("Generator/map.jpg", img)
        cv2.imshow("image", img)

    def noise(self):
        def riged(x, y):
            e0 = 1.00 * abs(noise.pnoise2(1 * x / self.rows * parameters["shape"]["scale"] * 8,
                                          1 * y / self.cols * parameters["shape"]["scale"] * 8)) * 1
            e1 = 0.50 * abs(noise.pnoise2(2 * x / self.rows * parameters["shape"]["scale"] * 8,
                                          2 * y / self.cols * parameters["shape"]["scale"] * 8)) * e0
            e2 = 0.25 * abs(noise.pnoise2(4 * x / self.rows * parameters["shape"]["scale"] * 8,
                                          4 * y / self.cols * parameters["shape"]["scale"] * 8)) * (e0 + e1)
            return e0 + e1 + e2

        img = np.zeros((self.rows, self.cols), np.uint8)
        for x in range(0, self.rows):
            for y in range(0, self.cols):
                e = riged(x, y)
                img[x][y] = int(e * 255)

        cv2.imshow("noise", img)


if __name__ == "__main__":
    m = MapGenerator(512, 512)
    m.noise()
    cv2.waitKey(1)
    m.generate()
    m.show_layer("water")
    m.show_layer("elevation")
    m.show_layer("moisture")
    m.show_layer("temperature")
    m.show()
    cv2.waitKey()
