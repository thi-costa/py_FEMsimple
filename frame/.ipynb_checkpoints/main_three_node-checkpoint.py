# Bibliotecas
import pandas as pd
import numpy as np
import numpy.linalg as nplin
import scipy.linalg as scilin
import sympy as sy
from scipy import linalg
import matplotlib.pyplot as plt
from scipy import linalg
import copy
import math
import frame_three_node as fr_th
from frame_three_node import Frame_euler
import time

'''
nodes = {1: (0, 0), 2: (0, 2), 3: (1, 3), 4: (2, 4), 5: (4, 4), 6: (4, 0)}
elems = {1: (1, 2), 2: (2, 3), 3: (3, 4), 4: (4, 5), 5: (5, 6)}
gl = {}
gl_fixed = {1: 0, 2: 0, 17: 0}
px_pred = {2: 1000}
qx_pred = {}
py_pred = {4: -1000}
qy_pred = {2: -1000, 3: -1000, 4: -1000}
m_pred = {}

I = {key: 0.36/100 for (key, value) in elems.items()}

E = {key: 210 * 10 ** 9 for (key, value) in elems.items()}

A = {key: 0.2*0.6 for (key, value) in elems.items()}
'''

nodes, elems, gl_fixed, px_pred, qx_pred, py_pred, qy_pred, m_pred, I, E, A = fr_th.read_frame()

frame_one = Frame_euler(nodes, elems, gl_fixed, px_pred, qx_pred, py_pred, qy_pred,
                  m_pred, I, E, A)

elem = 4
# print(f'Graus de liberdade do pórtico: {frame_one.gl}')
# print(f'Comprimento do elemento {elem}: {frame_one.points(elem)[2]}')
# print(f'Coseno x e y do elemento {elem}:', f'({frame_one.cos_dir(elem)[0]}, {frame_one.cos_dir(elem)[1]})')

#print(f'Matriz local do elemento {elem}:\n {frame_one.k_loc(elem)}')

#print(f'Força equivalente total (kN): {frame_one.f_equiv()[0]}')
#print(f'Força equivalente distribuida (kN): {frame_one.f_equiv()[2]["F. Dist. (kN/m)"]}')
#print(f'Força concentrada (kN): {frame_one.f_equiv()[2]["F. CC (kN)"]}')
#print(f'Matriz global:\n {frame_one.k_glob()[0][0:8, 0:8]}')


#print(f'Resultados de deslocamentos u: {frame_one.solver()}')

pontos = 5
#print(f'Resultados de normal:\n {frame_one.post_proc(pontos)[0]}', '\n')
# print(f'Resultados de fletor:\n {frame_one.post_proc(pontos)[1]}', '\n')
#print(f'Resultados de cortante:\n {frame_one.post_proc(pontos)[2]}')

frame_one.results_view()

time1 = time.perf_counter()
frame_one.post_proc(2)
time2 = time.perf_counter()

print(f'Cálculo de deslocamento e esforços (com aproximação de {pontos} pontos/elemento) demorou: {10**3*(time2-time1)} ms')
