#%%
from kivy.app import App
from kivy.animation import Animation
from kivy.config import Config
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.graphics import *
from kivy.clock import Clock
import numpy as np
import threading
import time

from deap import creator, base, tools, algorithms
import random
import multiprocessing

#%%
W, H = 400, 400
sz = (2, 100)
creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=sz[0] * sz[1] * 9)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_f(individual, data=False): # bit to float64
    result = []
    for p in range(0, len(individual), 9):
        # --- 9bit ---
        value = 0
        for i in individual[p : p + 9]:
            value = (value << 1) + i
        result.append(value / 1.3)
        # --- float64 ---
        # value = 0
        # for i in individual[p : p + 63]:
        #     value = (value << 1) + i
        # value = struct.unpack(">d", struct.pack(">q", value))[0]
        # result.append(value)
    h = np.reshape(np.array(result), sz)
    f = lambda h: -np.sum(np.abs(h.T - (H // 2, W // 2)))

    if data:
        return f(h), h
    return f(h)

pool = multiprocessing.Pool(processes=4)
toolbox.register("map", pool.map)
toolbox.register("evaluator", eval_f)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selection.selTournament, tournsize=5)

#%%
Config.set("graphics", "width", W)
Config.set("graphics", "height", H)

class AnimWidget(Widget):
    def __init__(self, interval):
        super().__init__()

        self.pops = 50
        self.population = toolbox.population(self.pops)
        Clock.schedule_interval(self.update, interval)
        self.epoch = 0

    def update(self, *args):
        self.canvas.clear()
        self.epoch += 1

        top10 = tools.selBest(self.population, k=1)
        elite = top10[0]

        offspring = algorithms.varAnd(self.population, toolbox, cxpb=1.0, mutpb=0.7)
        fits = list(toolbox.map(toolbox.evaluator, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = [fit]
        self.population = toolbox.select(offspring, k=self.pops)
        self.population[0] = elite

        top10 = tools.selBest(self.population, k=1)
        top_v, top = eval_f(top10[0], True)
        with self.canvas:
            Color(rgba=(1, 0.8, 0.1, 0.7))
            for x, y in zip(top[0], top[1]):
                Ellipse(pos=(x, y), size=(5, 5))
            self.label = Label(text="{:.2f} : {}".format(float(top_v), self.epoch), halign="right")

class Main(App):
    def build(self):
        return AnimWidget(0.03)


Main().run()
