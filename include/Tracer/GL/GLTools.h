#pragma once

#include <GL/glew.h>
#include <cstdio>
#include <cstring>

namespace gl
{
void CheckGL();

GLuint CreateVBO(const GLfloat *data, const unsigned int size);

void BindVBO(const unsigned int idx, const unsigned int N, const GLuint id);

void CheckFrameBuffer();

void CheckShader(GLuint shader);

void CheckProgram(GLuint id);

void DrawQuad();
} // namespace gl
