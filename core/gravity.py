import time

import taichi as ti
import numpy as np
from typing import List
from core.render import Renderer
from settings import camera_settings

ti.init(arch=ti.vulkan)


def timer(f):
    def wrapper(*args, **kwargs):
        st = time.time()
        ret = f(*args, **kwargs)
        # print(f"Passed for {f.__name__}:", time.time() - st)
        return ret
    return wrapper


@ti.data_oriented
class GravityEngine:
    TIME = 1
    ENABLED = False

    def __init__(self, n):
        self.n = n
        self.num_nodes = 2 * n - 1
        self.n_padded = 1 << (self.n - 1).bit_length() if self.n > 1 else 1

        # Входные данные
        self.raw_pos = ti.Vector.field(3, dtype=ti.f32, shape=n)
        self.raw_mass = ti.field(dtype=ti.f32, shape=n)
        self.velocities = ti.Vector.field(3, dtype=ti.f32, shape=n)

        # Поля для сортировки
        self.tmp_keys = ti.field(ti.u32, shape=self.n_padded)
        self.tmp_indices = ti.field(ti.i32, shape=self.n_padded)
        self.id_to_leaf = ti.field(dtype=ti.i32, shape=n)

        # Структура дерева
        self.morton_codes = ti.field(dtype=ti.u32, shape=n)
        self.masses = ti.field(dtype=ti.f32, shape=self.num_nodes)
        self.pos_cm = ti.Vector.field(3, dtype=ti.f32, shape=self.num_nodes)
        self.node_size = ti.field(dtype=ti.f32, shape=self.num_nodes)

        # Топология
        self.parent = ti.field(dtype=ti.i32, shape=self.num_nodes)
        self.left_child = ti.field(dtype=ti.i32, shape=self.num_nodes)
        self.right_child = ti.field(dtype=ti.i32, shape=self.num_nodes)
        self.atomic_counter = ti.field(dtype=ti.i32, shape=n - 1)

        # Результат
        self.acc = ti.Vector.field(3, dtype=ti.f32, shape=self.n)

    @ti.kernel
    def integrate_and_update(self, dt: ti.f32):
        limit = ti.cast(float(camera_settings.far), ti.f32)
        for i in range(self.n):
            self.velocities[i] += self.acc[i] * dt
            self.raw_pos[i] += self.velocities[i] * dt

            # Проверка границ для каждой оси (X, Y, Z)
            for d in ti.static(range(3)):
                if self.raw_pos[i][d] > limit:
                    self.raw_pos[i][d] = limit
                    self.velocities[i][d] *= -0.5  # Отскок с потерей энергии
                elif self.raw_pos[i][d] < -limit:
                    self.raw_pos[i][d] = -limit
                    self.velocities[i][d] *= -0.5

    def set_initial_state(self, np_pos, np_vel, np_mass):
        self.raw_pos.from_numpy(np_pos.astype(np.float32))
        self.velocities.from_numpy(np_vel.astype(np.float32))
        self.raw_mass.from_numpy(np_mass.astype(np.float32))

    @ti.func
    def count_leading_zeros(self, val: ti.u32) -> int:
        res = 0
        if val == 0:
            res = 32
        else:
            x = val
            if x >= 0x10000:
                x >>= 16; res += 0
            else:
                res += 16
            if x >= 0x100:
                x >>= 8
            else:
                res += 8
            if x >= 0x10:
                x >>= 4
            else:
                res += 4
            if x >= 0x4:
                x >>= 2
            else:
                res += 2
            if x >= 0x2:
                pass
            else:
                res += 1
        return res

    @ti.func
    def common_upper_bits(self, i: int, j: int):
        res = -1
        if 0 <= i < self.n and 0 <= j < self.n:
            c1, c2 = self.morton_codes[i], self.morton_codes[j]
            if c1 != c2:
                res = self.count_leading_zeros(c1 ^ c2)
            else:
                res = 32 + self.count_leading_zeros(ti.cast(i ^ j, ti.u32))
        return res

    @ti.kernel
    def _fast_reset(self):
        for i in range(self.num_nodes):
            self.parent[i] = -1
            self.masses[i] = 0.0
            if i < self.n - 1:
                self.atomic_counter[i] = 0

    @ti.kernel
    def build_topology(self):
        ti.loop_config(block_dim=256)
        for i in range(self.n - 1):
            d = 1 if self.common_upper_bits(i, i + 1) > self.common_upper_bits(i, i - 1) else -1
            delta_min = self.common_upper_bits(i, i - d)
            l_max = 2
            while self.common_upper_bits(i, i + d * l_max) > delta_min:
                l_max *= 2
            l = 0
            t = l_max
            while t > 1:
                t //= 2
                if self.common_upper_bits(i, i + d * (l + t)) > delta_min:
                    l += t
            j = i + d * l
            delta_node = self.common_upper_bits(i, j)
            s = 0
            t_split = l
            while t_split > 1:
                t_split = (t_split + 1) // 2
                if self.common_upper_bits(i, i + d * (s + t_split)) > delta_node:
                    s += t_split
            split = i + d * s + (d if d < 0 else 0)

            lc = self.n - 1 + split if min(i, j) == split else split
            rc = self.n - 1 + split + 1 if max(i, j) == split + 1 else split + 1

            self.left_child[i], self.right_child[i] = lc, rc
            self.parent[lc], self.parent[rc] = i, i

    @ti.kernel
    def aggregate_masses(self):
        ti.loop_config(block_dim=256)

        for i in range(self.n):
            curr = self.n - 1 + i
            while curr != 0:
                p = self.parent[curr]
                if p == -1: break
                if ti.atomic_add(self.atomic_counter[p], 1) == 0: break

                l, r = self.left_child[p], self.right_child[p]
                m_l, m_r = self.masses[l], self.masses[r]

                total_m = m_l + m_r
                self.masses[p] = total_m

                if total_m > 0:
                    self.pos_cm[p] = (self.pos_cm[l] * m_l + self.pos_cm[r] * m_r) / total_m
                self.node_size[p] = ti.max((self.pos_cm[p] - self.pos_cm[l]).norm() + self.node_size[l],
                                           (self.pos_cm[p] - self.pos_cm[r]).norm() + self.node_size[r])
                curr = p

    @ti.func
    def expand_bits(self, v: ti.u32) -> ti.u32:
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    @ti.kernel
    def _compute_morton_gpu(self, p_min: ti.types.vector(3, ti.f32), p_max: ti.types.vector(3, ti.f32)):
        span = p_max - p_min + 1e-6
        for i in range(self.n_padded):
            if i < self.n:
                norm = (self.raw_pos[i] - p_min) / span
                c = ti.cast(ti.max(0.0, ti.min(norm * 1023.0, 1023.0)), ti.u32)
                self.tmp_keys[i] = self.expand_bits(c[0]) | (self.expand_bits(c[1]) << 1) | (
                            self.expand_bits(c[2]) << 2)
            else:
                self.tmp_keys[i] = ti.u32(0xFFFFFFFF)
            self.tmp_indices[i] = i

    @ti.kernel
    def _bitonic_sort_step(self, i_step: ti.i32, j_step: ti.i32):
        ti.loop_config(block_dim=512)
        for i in range(self.n_padded):
            dist = 1 << j_step
            p2 = i ^ dist
            if p2 > i:
                direction = (i >> (i_step + 1)) & 1
                # Прямое обращение к полям без лишних переменных
                if (self.tmp_keys[i] > self.tmp_keys[p2]) == (direction == 0):
                    self.tmp_keys[i], self.tmp_keys[p2] = self.tmp_keys[p2], self.tmp_keys[i]
                    self.tmp_indices[i], self.tmp_indices[p2] = self.tmp_indices[p2], self.tmp_indices[i]

    def _gpu_argsort_internal(self):
        steps = int(np.log2(self.n_padded))
        for i in range(steps):
            for j in range(i, -1, -1):
                self._bitonic_sort_step(i, j)

    @ti.kernel
    def get_bounds(self) -> ti.types.vector(6, ti.f32):
        p_min = ti.Vector([1e10, 1e10, 1e10])
        p_max = ti.Vector([-1e10, -1e10, -1e10])
        for i in range(self.n):
            ti.atomic_min(p_min, self.raw_pos[i])
            ti.atomic_max(p_max, self.raw_pos[i])
        return ti.Vector([p_min[0], p_min[1], p_min[2], p_max[0], p_max[1], p_max[2]])

    @ti.kernel
    def _prepare_tree_data_gpu(self):
        for i in range(self.n):
            # tmp_indices после сортировки хранит оригинальный индекс тела,
            # которое должно оказаться на i-й позиции в массиве листьев
            old_idx = self.tmp_indices[i]
            leaf_idx = (self.n - 1) + i

            # Записываем Morton-код (нужен для build_topology)
            self.morton_codes[i] = self.tmp_keys[i]

            # Переносим физические данные в листья дерева
            self.pos_cm[leaf_idx] = self.raw_pos[old_idx]
            self.masses[leaf_idx] = self.raw_mass[old_idx]

            # КРИТИЧЕСКИЙ МОМЕНТ: связываем оригинальный ID с индексом в дереве
            # Чтобы в compute_forces знать, кто есть кто
            self.id_to_leaf[old_idx] = leaf_idx

    def build_internal(self):
        # 1. Считаем границы системы (GPU)
        bounds = self.get_bounds()
        p_min = ti.Vector([bounds[0], bounds[1], bounds[2]])
        p_max = ti.Vector([bounds[3], bounds[4], bounds[5]])

        # 2. Генерируем Morton-коды (GPU)
        self._compute_morton_gpu(p_min, p_max)

        # 3. Сортируем данные (GPU - Bitonic Sort)
        # Это заменяет медленный np.argsort
        self._gpu_argsort_internal()

        # 4. Сбрасываем структуру дерева (GPU)
        self._fast_reset()

        # 5. Заполняем листья и карту id_to_leaf (GPU)
        self._prepare_tree_data_gpu()

        # 6. Строим топологию и агрегируем массы (GPU)
        self.build_topology()
        self.aggregate_masses()
        # self.debug_print()

    def debug_print(self):
        parents = self.parent.to_numpy()
        lefts = self.left_child.to_numpy()
        rights = self.right_child.to_numpy()
        masses = self.masses.to_numpy()
        # Вытягиваем векторное поле
        pos_cm = self.pos_cm.to_numpy()

        print(f"\n{'=' * 95}")
        print(
            f"{'INDEX':<6} | {'TYPE':<8} | {'PARENT':<6} | "
            f"{'L/R CHILD':<12} | {'MASS':<12} | {'CENTER OF MASS (X, Y, Z)':<30}"
        )
        print(f"{'-' * 95}")

        for i in range(self.num_nodes):
            node_type = "INT" if i < self.n - 1 else "LEAF"
            p = parents[i]
            l_r = f"{lefts[i] if i < self.n - 1 else '-'}/{rights[i] if i < self.n - 1 else '-'}"
            m = masses[i]
            pos = pos_cm[i]
            pos_str = f"({pos[0]:>8.1f}, {pos[1]:>8.1f}, {pos[2]:>8.1f})"

            # Подсветка для тяжелых объектов (Земля/Луна)
            marker = " <--" if m > 1e20 else ""

            print(f"{i:<6} | {node_type:<8} | {p:<6} | {l_r:<12} | {m:<12.2e} | {pos_str}{marker}")
        print(f"{'=' * 95}\n")

    @ti.kernel
    def compute_forces(self):
        ti.loop_config(block_dim=64)
        softening_sq = 100.0
        theta_sq = 0.49
        G = 6.674e-20

        for i in range(self.n):
            pos_i = self.raw_pos[i]
            my_leaf_idx = self.id_to_leaf[i]

            acc_i = ti.Vector([0.0, 0.0, 0.0])
            stack = ti.Vector([0] * 64)
            stack_ptr = 1  # Корень
            while stack_ptr > 0:
                stack_ptr -= 1
                curr = stack[stack_ptr]

                diff = self.pos_cm[curr] - pos_i
                dist_sq = diff.norm_sqr() + softening_sq

                # Если это лист ИЛИ узел достаточно далеко
                if curr >= self.n - 1 or self.node_size[curr] ** 2 < dist_sq * theta_sq:
                    # Проверка: не считаем силу от самого себя (если это лист)
                    if self.masses[curr] > 0 and curr != my_leaf_idx:
                        inv_dist = 1.0 / ti.sqrt(dist_sq)
                        acc_i += diff * (self.masses[curr] * G * inv_dist ** 3)
                else:
                    # Раскрываем узел
                    stack[stack_ptr] = self.left_child[curr]
                    stack[stack_ptr + 1] = self.right_child[curr]
                    stack_ptr += 2
            self.acc[i] = acc_i

    def update_physics(self, renderers: List[Renderer], dt):
        if not self.ENABLED:
            return
        dt_val = dt * self.TIME

        # 1-3. Физика на GPU
        self.build_internal()
        self.compute_forces()
        self.integrate_and_update(dt_val)

        # 4. Выгрузка без ошибок
        # Используем одномерный массив, чтобы Taichi не спотыкался об размерности
        full_pos = self.raw_pos.to_numpy()

        offset = 0
        for i, renderer in enumerate(renderers):
            n_obj = renderer.instance_count
            # Срезаем нужную часть данных
            current_batch = full_pos[offset: offset + n_obj]

            # Отправляем в OpenGL
            renderer.update_positions(current_batch)
            offset += n_obj
