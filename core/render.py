import os

import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader


class Renderer:
    def __init__(self, model, scale=1.0, positions=None, is_point=False, material_config=None):
        self.instance_count = len(positions) if positions is not None else 1
        self.scale = scale
        self.is_point = is_point
        self.shader = self._compile_shaders()

        self.pos_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        glBufferData(GL_ARRAY_BUFFER, positions.nbytes, positions, GL_DYNAMIC_DRAW)

        # Кешируем локации униформ
        self.locs = {
            "view": glGetUniformLocation(self.shader, "view"),
            "proj": glGetUniformLocation(self.shader, "projection"),
            "scale": glGetUniformLocation(self.shader, "scale"),
            "matType": glGetUniformLocation(self.shader, "matType"),
            "texDay": glGetUniformLocation(self.shader, "texDay"),
            "texNight": glGetUniformLocation(self.shader, "texNight")
        }

        self.mesh_parts = []

        for part_name, data in model.parts.items():
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            # Геометрия
            vertices = np.hstack((data['v'], data['uv'])).astype(np.float32)
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * 4, ctypes.c_void_p(12))
            glEnableVertexAttribArray(1)

            # ПРИВЯЗЫВАЕМ ТОТ ЖЕ САМЫЙ pos_vbo К ЭТОМУ VAO
            glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)  # Используем общий буфер
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(2)
            glVertexAttribDivisor(2, 1)

            # Настройка материалов и текстур
            conf = material_config.get(part_name, {}) if material_config else {}
            mat_type = conf.get('type', 0)

            # Грузим текстуры
            tex_day = self._load_texture(os.path.join(model.path, conf.get('day', f"{part_name.lower()}.jpg")))
            tex_night = 0
            if mat_type == 2:  # Если Земля
                tex_night = self._load_texture(os.path.join(model.path, conf.get('night', 'Night_lights_2K.jpg')))

            self.mesh_parts.append({
                'vao': vao,
                'count': len(data['v']),
                'texDay': tex_day,
                'texNight': tex_night,
                'type': mat_type
            })

    @staticmethod
    def _compile_shaders():
        v_src = """
            #version 330
            layout(location = 0) in vec3 pos;
            layout(location = 1) in vec2 tex;
            layout(location = 2) in vec3 instPos;

            uniform mat4 view;
            uniform mat4 projection;
            uniform float scale;
            uniform vec3 cameraPos;

            out vec2 v_tex;
            out vec3 v_normal;
            out vec3 v_lightDir;

            void main() {
                v_tex = tex;
                // Упрощенный вектор света (пусть светит из центра системы - от Солнца)
                v_lightDir = normalize(vec3(0.0) - instPos); 
                v_normal = normalize(pos); // Для сферы нормаль совпадает с локальной позицией

                vec3 finalPos = pos * scale + instPos;
                gl_Position = projection * view * vec4(finalPos, 1.0);
            }
            """
        f_src = """
            #version 330
            in vec2 v_tex;
            in vec3 v_normal;
            in vec3 v_lightDir;

            uniform sampler2D texDay;    // Основная (Diffuse / Clouds)
            uniform sampler2D texNight;  // Ночные огни
            uniform int matType;         // 0: Normal, 1: Transparent(Clouds), 2: Earth(Day/Night)

            out vec4 FragColor;

            void main() {
                if (matType == 2) { // Логика ЗЕМЛИ
                    float diff = dot(normalize(v_normal), normalize(v_lightDir));
                    vec4 day = texture(texDay, v_tex);
                    vec4 night = texture(texNight, v_tex);

                    // Плавный переход дня в ночь
                    float intensity = clamp(diff * 2.0 + 0.5, 0.0, 1.0);
                    FragColor = mix(night, day, intensity);

                } else if (matType == 1) { // Логика ОБЛАКОВ
                    vec4 color = texture(texDay, v_tex);
                    float alpha = color.r; // Черный = прозрачный
                    if (alpha < 0.05) discard;
                    FragColor = vec4(color.rgb, alpha);

                } else { // Обычный режим (Атмосфера или другие планеты)
                    FragColor = texture(texDay, v_tex);
                }
            }
            """
        return compileProgram(
            compileShader(v_src, GL_VERTEX_SHADER),
            compileShader(f_src, GL_FRAGMENT_SHADER)
        )

    def render(self, view_mat, proj_mat):
        glUseProgram(self.shader)
        glUniformMatrix4fv(self.locs["view"], 1, GL_FALSE, view_mat)
        glUniformMatrix4fv(self.locs["proj"], 1, GL_FALSE, proj_mat)
        glUniform1f(self.locs["scale"], self.scale)

        # Включаем прозрачность (нужно для облаков/атмосферы)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_PROGRAM_POINT_SIZE)

        if self.is_point:
            glPointSize(2.0)  # Сделай чуть больше 1.0, чтобы на 4К мониторах не пропали
            # ВАЖНО: Нужно привязать VAO хотя бы одной части, чтобы заработал атрибут instPos (location 2)
            glBindVertexArray(self.mesh_parts[0]['vao'])
            glDrawArraysInstanced(GL_POINTS, 0, 1, self.instance_count)
            glBindVertexArray(0)
            return

        for part in self.mesh_parts:
            glUniform1i(self.locs["matType"], part['type'])

            # Слот 0: День
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, part['texDay'])
            glUniform1i(self.locs["texDay"], 0)

            # Слот 1: Ночь
            if part['type'] == 2:
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, part['texNight'])
                glUniform1i(self.locs["texNight"], 1)

            glBindVertexArray(part['vao'])
            glDrawArraysInstanced(GL_TRIANGLES, 0, part['count'], self.instance_count)

        glDisable(GL_BLEND)

    @staticmethod
    def _load_texture(path):
        if not os.path.exists(path):
            return 0
        img = pygame.image.load(path).convert_alpha()
        raw = pygame.image.tostring(img, "RGBA", True)
        tid = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tid)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.get_width(), img.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, raw)
        glGenerateMipmap(GL_TEXTURE_2D)
        return tid

    def update_positions(self, new_positions):
        glBindBuffer(GL_ARRAY_BUFFER, self.pos_vbo)
        # Используем nbytes, чтобы точно знать размер данных
        glBufferSubData(GL_ARRAY_BUFFER, 0, new_positions.nbytes, new_positions)
