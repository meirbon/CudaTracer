#include <Tracer/GL/GLTools.h>
#include <iostream>

namespace gl
{
	void CheckGL()
	{
		GLenum error = glGetError();
		if (error != GL_NO_ERROR)
		{
			char t[1024];
			sprintf(t, "error %i (%x)\n", error, error);
			if (error == 0x500)
				strcat(t, "INVALID ENUM");
			else if (error == 0x502)
				strcat(t, "INVALID OPERATION");
			else if (error == 0x501)
				strcat(t, "INVALID VALUE");
			else if (error == 0x506)
				strcat(t, "INVALID FRAMEBUFFER OPERATION");
			else
				strcat(t, "UNKNOWN ERROR");
			std::cout << "OpenGL Error" << t;
		}
	}

	GLuint CreateVBO(const GLfloat *data, const unsigned int size)
	{
		GLuint id;
		glGenBuffers(1, &id);
		glBindBuffer(GL_ARRAY_BUFFER, id);
		glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
		return id;
	}

	void BindVBO(const unsigned int idx, const unsigned int N, const GLuint id)
	{
		glEnableVertexAttribArray(idx);
		glBindBuffer(GL_ARRAY_BUFFER, id);
		glVertexAttribPointer(idx, N, GL_FLOAT, GL_FALSE, 0, (void *)0);
	}

	void CheckFrameBuffer()
	{
		if (glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
			return;
		std::cout << "Incomplete framebuffer" << std::endl;
		exit(1);
	}

	void CheckShader(GLuint shader)
	{
		char buffer[1024];
		memset(buffer, 0, 1024);
		GLsizei length = 0;
		GLint status;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
		if (status != GL_TRUE) // thanks Rik
		{
			glGetShaderInfoLog(shader, 1024, &length, buffer);
			std::cout << "Shader compile error: " << buffer << std::endl;
		}
	}

	void CheckProgram(GLuint id)
	{
		char buffer[1024];
		memset(buffer, 0, 1024);
		GLsizei length = 0;
		glGetProgramInfoLog(id, 1024, &length, buffer);

		if (length > 0)
		{
			if (strstr(buffer, "WARNING"))
				std::cout << "Shader link error: " << buffer << std::endl;
			else if (!strstr(buffer, "No errors"))
				std::cout << "Shader link error: " << buffer << std::endl;
		}
	}

	void DrawQuad()
	{
		static GLuint vao = 0;
		if (!vao)
		{
			// generate buffers
			GLfloat verts[] = { -1, -1, 0, 1, -1, 0, -1, 1, 0, 1, -1, 0, -1, 1, 0, 1, 1, 0 };
			GLfloat uvdata[] = { 0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0 };
			GLuint vertexBuffer = CreateVBO(verts, sizeof(verts));
			GLuint UVBuffer = CreateVBO(uvdata, sizeof(uvdata));
			glGenVertexArrays(1, &vao);

			glBindVertexArray(vao);
			BindVBO(0, 3, vertexBuffer);
			BindVBO(1, 2, UVBuffer);
			glBindVertexArray(0);
			glDisableVertexAttribArray(0);
			glDisableVertexAttribArray(1);
		}
		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, 6);
		glBindVertexArray(0);
	}
} // namespace gl