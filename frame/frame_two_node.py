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

# Frame class
class Frame:

    def __init__(self, nodes, elems, gl_fixed, px, qx, py, qy, m, I, E, A, v, beam_type, factor=np.float(5/6)):
        self.nodes = nodes
        self.gl = {key: (key * 3 - 2, key * 3 - 1, key * 3) for (key, value) in nodes.items()}
        self.elems = elems
        self.gl_fixed = gl_fixed
        self.px = px
        self.qx = qx
        self.py = py
        self.qy = qy
        self.m = m
        self.I = I
        self.E = E
        self.A = A
        self.v = v
        phi = {}
        self.phi = phi
        self.beam_type = beam_type
        self.factor = factor
        self.x_axis = np.array([1, 0])
        self.y_axis = np.array([0, 1])
        self.ngl = len(nodes) * 3

    def points(self, elem):
        '''
        Retorna as coordenadas do ponto 0 e do ponto 1 de um elemento
        e seu comprimento
        '''
        elements = self.elems
        nodes = self.nodes

        # Nós que o elemento se conecta
        no_0 = elements[elem][0]
        no_1 = elements[elem][1]

        # Coordenadas de cada nó e comprimento do elemento
        p_0 = np.array(nodes[no_0])
        p_1 = np.array(nodes[no_1])
        L = nplin.norm(p_1 - p_0)

        return p_0, p_1, L

    def cos_dir(self, elem):
        '''
        Retorna os cosenos diretores x e y do elemento
        '''

        p_0, p_1, L = self.points(elem)
        c_x = ((p_1 - p_0) / L)[0]
        c_y = ((p_1 - p_0) / L)[1]

        return c_x, c_y

    def k_loc(self, elem):
        """
        Retorna a matriz K local do elemento
        """
        
        if self.beam_type=="euler":
            p_0, p_1, L = self.points(elem)
            E = self.E[elem]
            I = self.I[elem]
            A = self.A[elem]

            k_l = np.array([[E * A / L, 0, 0, -E * A / L, 0, 0],
                            [0, 12 * E * I / L ** 3, 6 * E * I / L ** 2, 0, -12 * E * I / L ** 3, 6 * E * I / L ** 2],
                            [0, 6 * E * I / L ** 2, 4 * E * I / L, 0, -6 * E * I / L ** 2, 2 * E * I / L],
                            [-E * A / L, 0, 0, E * A / L, 0, 0],
                            [0, -12 * E * I / L ** 3, -6 * E * I / L ** 2, 0, 12 * E * I / L ** 3, -6 * E * I / L ** 2],
                            [0, 6 * E * I / L ** 2, 2 * E * I / L, 0, -6 * E * I / L ** 2, 4 * E * I / L]])


            return k_l
        
        elif self.beam_type == "timo":
            p_0, p_1, L = self.points(elem)
            E = self.E[elem]
            I = self.I[elem]
            A = self.A[elem]
            v = self.v[elem]
            phi = self.phi
            factor = self.factor
            
            G = E / ( 2 * ( 1 + v))
            g = (E*I) / (A * factor * G)
            phi = (12 * g) / (L**2)
            
            k_l_aux = ((E * I) / (L**3 * (1+phi))
                   *np.array([[12,             6*L,  -12,            6*L],
                              [6*L, (4+phi)*L**2, -6*L, (2-phi)*L**2],
                              [-12,           -6*L,   12,           -6*L],
                              [6*L, (2-phi)*L**2, -6*L, (4+phi)*L**2]]))
            
            k_l = np.array([[E * A / L, 0, 0, -E * A / L, 0, 0],
                            [0, k_l_aux[0][0], k_l_aux[0][1], 0, k_l_aux[0][2], k_l_aux[0][3]],
                            [0, k_l_aux[1][0], k_l_aux[1][1], 0, k_l_aux[1][2], k_l_aux[1][3]],
                            [-E * A / L, 0, 0, E * A / L, 0, 0],
                            [0, k_l_aux[2][0], k_l_aux[2][1], 0, k_l_aux[2][2], k_l_aux[2][3]],
                            [0, k_l_aux[3][0], k_l_aux[3][1], 0, k_l_aux[3][2], k_l_aux[3][3]]])
            
            return k_l

    def rot_mat(self, elem):
        '''
        Retorna matriz de rotação de um elemento
        '''
        l, m = self.cos_dir(elem)
        r = np.array([[l, m, 0, 0, 0, 0],
                      [-m, l, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, l, m, 0],
                      [0, 0, 0, -m, l, 0],
                      [0, 0, 0, 0, 0, 1]])

        return r

    def k_glob(self):
        '''
        Retorna 3 objetos:
            . Matriz (numpy) de rigidez global;
            . Matriz (numpy) de rigidez global simplificada;
            . Uma lista de matrizes globais para cada elemento.
        '''
        # Dados necessários
        elements = self.elems
        ngl = self.ngl
        gl_fixed = list(self.gl_fixed.keys())
        ke_list = []
        K = np.zeros((ngl, ngl))

        for elem in elements:
            # Transformar matriz local em global
            no_0, no_1 = elements[elem][0], elements[elem][1]

            tau = self.rot_mat(elem)
            k_r = tau.T.dot(self.k_loc(elem)).dot(tau)

            K_rG = np.zeros((ngl, ngl))
            ind = [3 * no_0 - 3, 3 * no_0 - 2, 3 * no_0 - 1,
                   3 * no_1 - 3, 3 * no_1 - 2, 3 * no_1 - 1]

            for i in range(6):
                for j in range(6):
                    K_rG[ind[i], ind[j]] = k_r[i, j]

            ke_list.append(K_rG)
            K += K_rG

        # Remover graus de liberdade restringidos
        remove_ind = np.array(np.array(gl_fixed) - 1, 'int')
        K_orig = copy.deepcopy(K)

        for i in [0, 1]:
            K = np.delete(K, remove_ind, axis=i)

        writer = pd.ExcelWriter('m_matrix.xlsx', engine='xlsxwriter')

        pd.DataFrame(K).to_excel(writer, sheet_name='k_simp', index=False)

        writer.save()

        return K_orig, K, ke_list

    def f_equiv(self):
        '''
        Retorna vetor de forças equivalente totais e considerando
        os graus de liberdade fixos
        '''
        # Dados necessários
        elements = self.elems
        ngl = self.ngl
        qx, px = self.qx, self.px
        qy, py = self.qy, self.py
        m = self.m
        nodes = self.nodes
        gl_fixed = list(self.gl_fixed.keys())
        f_descr = {}
        f_q = np.zeros((ngl))
        f_cc = np.zeros((ngl))

        for elem in elements.keys():
            L = self.points(elem)[2]
            no_0, no_1 = elements[elem][0], elements[elem][1]

            if elem in qx.keys():
                qs = qx[elem]
            else:
                qs = 0

            if elem in qy.keys():
                qt = qy[elem]

            else:
                qt = 0

            #fq_e = np.array([1 / 2 * qs * L,
             #                1 / 2 * qt * L,
              #               1 / 12 * qt * L ** 2,
               #              1 / 2 * qs * L,
                #             1 / 2 * qt * L,
                 #            -1 / 12 * qt * L ** 2])

            if qt != 0:
                fq_e = np.array([1 / 2 * qs * L,
                                 (L/20) * (7*qt[0] + 3*qt[1]),
                                 (L/20) * (L/3) * (3*qt[0] + 2*qt[1]),
                                 1 / 2 * qs * L,
                                 (L/20) * (3*qt[0] + 7*qt[1]),
                                 -(L/20) * (L/3) * (2*qt[0] + 3*qt[1])])
            else:
                fq_e = np.array([1 / 2 * qs * L,
                                 0,
                                 0,
                                 1 / 2 * qs * L,
                                 0,
                                 0])

            tau = self.rot_mat(elem)

            fq_e_aux = tau.T.dot(fq_e)

            ind = [3 * no_0 - 3, 3 * no_0 - 2, 3 * no_0 - 1,
                   3 * no_1 - 3, 3 * no_1 - 2, 3 * no_1 - 1]

            for i in range(6):
                f_q[ind[i]] = f_q[ind[i]] + fq_e_aux[i]

        for node in nodes:
            if node in px.keys():
                f_cc[node * 3 - 3] = px[node]

            if node in py.keys():
                f_cc[node * 3 - 2] = py[node]

            if node in m.keys():
                f_cc[node * 3 - 1] = m[node]

        f_descr['F. Dist. (kN/m)'] = f_q
        f_descr['F. CC (kN)'] = f_cc

        f = f_q + f_cc

        # Remover graus de liberdade restringidos
        remove_ind = np.array(np.array(gl_fixed) - 1, 'int')
        f_tot = copy.deepcopy(f)
        f = np.delete(f, remove_ind, axis=0)

        return f_tot, f, f_descr

    def solver(self):
        '''
        Retorna os resultados de deslocamento e reações
        '''

        # Dados de entrada
        K_orig, K, _ = self.k_glob()
        ngl = self.ngl
        F_tot, F, _ = self.f_equiv()
        gl_fixed_info = self.gl_fixed
        gl_fixed = np.array(list(gl_fixed_info.keys()), 'int') - 1
        gl_desloc = np.array(list(gl_fixed_info.values()))

        # Gls e gl desconhecidos
        gl = np.array([i for i in range(ngl)])
        gl_unknow = np.delete(gl, np.array(gl_fixed, 'int'))

        k_bckup = copy.deepcopy(K_orig)
        K_orig = np.delete(K_orig, gl_fixed, axis=0)
        K_orig = np.delete(K_orig, gl_unknow, axis=1)

        ku_presc = K_orig.dot(gl_desloc)

        # Vetor deslocamento e de reações
        u = np.zeros((self.ngl))
        u[gl_fixed] = gl_desloc

        # Adicionar direções consideradas fixas
        u[gl_unknow] = np.linalg.inv(K).dot(F - ku_presc)

        # Reações
        reac = np.dot(k_bckup, u) - F_tot.T

        return u, reac

    def post_proc(self, points):
        '''
        Retorna resultados de esforços com determinada quantidade
        de pontos, por elemento

        :return normal
        :return fletor
        :return cortante
        '''

        N = {}
        M = {}
        Q = {}

        # Dados de entrada
        elements = self.elems
        nodes = self.nodes
        E = self.E
        I = self.I
        A = self.A
        u = self.solver()[0]

        for elem in elements.keys():
            p_0, p_1, L = self.points(elem)
            no_0, no_1 = elements[elem][0], elements[elem][1]
            ind = [3 * no_0 - 3, 3 * no_0 - 2, 3 * no_0 - 1,
                   3 * no_1 - 3, 3 * no_1 - 2, 3 * no_1 - 1]

            # Deslocamentos em coordenadas locais
            u_loc = self.rot_mat(elem).dot(u[ind])
            v_1 = np.array([1 for i in range(points)])

            # Normal
            N[elem] = (E[elem] * A[elem]) * (
                    (-1 / L) * u_loc[0] + (1 / L) * u_loc[3]) * v_1

            # Fletor e Cortante
            ksi_v = np.linspace(-1, 1, max(points, 2))

            # Fletor
            m_i = np.zeros((4, points))
            c_m = (E[elem] * I[elem]) * (2 / L) ** 2

            m_i[0] = c_m * (3 * ksi_v / 2)
            m_i[1] = c_m * (L / 4 * (3 * ksi_v - 1))
            m_i[2] = c_m * (-3 * ksi_v / 2)
            m_i[3] = c_m * (L / 4 * (3 * ksi_v + 1))

            m_i = m_i.T.dot(u_loc[[1, 2, 4, 5]])
            M[elem] = m_i

            # Cortante
            v_i = np.zeros((4, points))
            c_v = (E[elem] * I[elem]) * (2 / L) ** 3

            v_i[0] = c_v * (3 / 2 * v_1)
            v_i[1] = c_v * (3*L / 4 * v_1)
            v_i[2] = c_v * (-3 / 2 * v_1)
            v_i[3] = c_v * (3*L / 4 * v_1)

            v_i = v_i.T.dot(u_loc[[1, 2, 4, 5]])
            Q[elem] = v_i

        return N, M, Q

    def results_view(self, path='results_2node.xlsx'):
        '''
        Imprime relatório de resolução do problema
        '''
        print('Resolução do pórtico - Exemplo\n')
        nodes = self.nodes
        elems = self.elems
        A = self.A
        E = self.E
        I = self.I
        gl_fixed = self.gl_fixed
        saida = {}


        print(f'NÚMERO DE NÓS: {len(nodes)}')
        print(f'NÚMERO DE ELEMENTOS: {len(elems)}\n')
        print('COORDENADAS DOS NÓS \n')

        dt_coords = pd.DataFrame()
        for no in nodes:
            dt_coords.loc[no - 1, 'NÓ'] = int(no)
            dt_coords.loc[no - 1, 'X'] = nodes[no][0]
            dt_coords.loc[no - 1, 'Y'] = nodes[no][1]

        print(dt_coords.set_index('NÓ'), '\n')
        saida['Coords'] = dt_coords

        dt_elems = pd.DataFrame()
        for elem in elems:
            dt_elems.loc[elem-1, 'ELEM'] = int(elem)
            dt_elems.loc[elem-1, 'NÓ_1'] = elems[elem][0]
            dt_elems.loc[elem - 1, 'NÓ_2'] = elems[elem][1]
            dt_elems.loc[elem-1, 'E'] = E[elem]
            dt_elems.loc[elem-1, 'A'] = A[elem]
            dt_elems.loc[elem-1, 'I'] = I[elem]

        print(dt_elems.set_index('ELEM'), '\n')
        saida['Elems'] = dt_elems

        print(f'NÚMERO DE CONDIÇÕES DE CONTORNO: {len(nodes) * 3}', '\n')

        dt_cc = pd.DataFrame()
        cont = 0
        for gl in gl_fixed.keys():
            dt_cc.loc[cont, 'GL'] = gl
            dt_cc.loc[cont, 'CONDIÇÃO DE CONTORNO'] = gl_fixed[gl]
            cont = cont + 1

        print(dt_cc.set_index('GL'), '\n')
        saida['CC'] = dt_cc

        dt_desloc, dt_reac = pd.DataFrame(), pd.DataFrame()
        u, reac = self.solver()
        print('DESLOCAMENTOS')
        # print(no_fix)
        for no in nodes.keys():
            dt_desloc.loc[no - 1, 'NÓ'] = no
            dt_desloc.loc[no - 1, 'DESLOC. X'] = np.round(u[no * 3 - 3], 12)
            dt_desloc.loc[no - 1, 'DESLOC. Y'] = np.round(u[no * 3 - 2], 12)
            dt_desloc.loc[no - 1, 'ROTAÇÃO'] = np.round(u[no * 3 - 1], 12)

        print(dt_desloc.set_index('NÓ'), '\n')
        saida['u_results'] = dt_desloc

        print('REAÇÕES DE APOIO')
        for no in nodes.keys():
            dt_reac.loc[no - 1, 'NÓ'] = no
            dt_reac.loc[no - 1, 'REAÇÃO X'] = np.round(reac[no * 3 - 3], 8)
            dt_reac.loc[no - 1, 'REAÇÃO Y'] = np.round(reac[no * 3 - 2], 8)
            dt_reac.loc[no - 1, 'REAÇÃO MOMENTO'] = np.round([no * 3 - 1], 8)


        print(dt_reac.set_index('NÓ'), '\n')
        saida['reactions_results'] = dt_reac

        print('ESFORÇOS INTERNOS')
        dt_esf = pd.DataFrame()
        node_list = list(nodes.keys())

        N, M, Q = self.post_proc(2)

        cont = 0
        e = 1
        for no in node_list:
            if no == node_list[0]:
                dt_esf.loc[cont, 'NÓ'] = no
                dt_esf.loc[cont, 'Posição'] = '-'
                dt_esf.loc[cont, 'Normal (N)'] = np.round(N[e][0], 4)
                dt_esf.loc[cont, 'Cortante (N)'] = np.round(Q[e][0], 4)
                dt_esf.loc[cont, 'Fletor (N)'] = np.round(M[e][0], 4)

                cont = cont + 1

            elif no != node_list[-1]:
                dt_esf.loc[cont, 'NÓ'] = no
                dt_esf.loc[cont, 'Posição'] = 'Esquerda'
                dt_esf.loc[cont, 'Normal (N)'] = np.round(N[e][-1], 4)
                dt_esf.loc[cont, 'Cortante (N)'] = np.round(Q[e][-1], 4)
                dt_esf.loc[cont, 'Fletor (N)'] = np.round(M[e][-1], 4)
                cont = cont + 1

                e = e + 1

                dt_esf.loc[cont, 'NÓ'] = no
                dt_esf.loc[cont, 'Posição'] = 'Direita'
                dt_esf.loc[cont, 'Normal (N)'] = np.round(N[e][0], 4)
                dt_esf.loc[cont, 'Cortante (N)'] = np.round(Q[e][0], 4)
                dt_esf.loc[cont, 'Fletor (N)'] = np.round(M[e][0], 4)
                cont = cont + 1
            else:
                dt_esf.loc[cont, 'NÓ'] = no
                dt_esf.loc[cont, 'Posição'] = '-'
                dt_esf.loc[cont, 'Normal (N)'] = np.round(N[e][-1], 4)
                dt_esf.loc[cont, 'Cortante (N)'] = np.round(Q[e][-1], 4)
                dt_esf.loc[cont, 'Fletor (N)'] = np.round(M[e][-1], 4)

        print(dt_esf, '\n')
        saida['stresses_results'] = dt_esf

        writer = pd.ExcelWriter(path, engine='xlsxwriter')
        for key, _ in saida.items():
            saida[key].to_excel(writer, sheet_name=key, index=False)

        writer.save()


def read_frame(b_type,
        path="./exemplo.xlsx"):
    # Variáveis de entrada para armazenar dados do problema
    nodes = {}
    elems = {}
    gl_fixed = {}
    px_pred = {}
    qx_pred = {}
    py_pred = {}
    qy_pred = {}
    m_pred = {}
    I = {}
    E = {}
    A = {}
    v = {}
    beam_type = b_type

    # Arquivo de leitura Excel
    frame_read = pd.read_excel(path, sheet_name=None)

    # Dados por nó
    for index, row in frame_read['node'].iterrows():
        nodes[int(row[0])] = (row[1], row[2])  # Coordenadas dos nós

    # Dados por elemento
    for index, row in frame_read['elems'].iterrows():
        elems[int(row[0])] = (int(row[1]), int(row[2]))  # Conectividade
        A[int(row[0])] = row[3]  # Área da seção
        E[int(row[0])] = row[4]  # Módulo de elasticidade do material
        I[int(row[0])] = row[5]  # Inércia
        v[int(row[0])] = row[6]  # Coeficiente de Poisson

    # Dados de força
    # Forças concentradas
    for index, row in frame_read['px_pred'].iterrows():  # px
        px_pred[int(row[0])] = row[1]

    for index, row in frame_read['py_pred'].iterrows():  # py
        py_pred[int(row[0])] = row[1]

    for index, row in frame_read['m_pred'].iterrows():  # m
        m_pred[int(row[0])] = row[1]

    # Forças distribuídas
    for index, row in frame_read['qx_pred'].iterrows():  # qx
        qx_pred[int(row[0])] = row[1]

    for index, row in frame_read['qy_pred'].iterrows():  # qy
        qy_pred[int(row[0])] = (row[1], row[2])

    # Graus de liberdade fixos
    for index, row in frame_read['gl_fixed'].iterrows():
        gl_fixed[int(row[0])] = row[1]

    return nodes, elems, gl_fixed, px_pred, qx_pred, py_pred, qy_pred, m_pred, I, E, A, v, beam_type
