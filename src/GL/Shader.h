#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include <GL/glew.h>
#include <glm/glm.hpp>

#include "GL/Texture.h"

namespace gl
{
	class ComputeBuffer;

	class Shader
	{
	public:
		// constructor / destructor
		Shader(const char *vfile, const char *pfile);

		~Shader();

		// get / set
		unsigned int GetID() { return m_Id; }

		// methods
		void Init(const char *vfile, const char *pfile);

		void Compile(const char *vtext, const char *ftext);

		void Bind();

		void SetInputTexture(unsigned int slot, const char *name, gl::Texture *texture);

		void SetInputMatrix(const char *name, const glm::mat4 &matrix);

		void Unbind();

		GLint getUniformLocation(const char *name);

		void setUniform1f(const char *name, float value);

		void setUniform1i(const char *name, int value);

		void setUniform2f(const char *name, glm::vec2 value);

		void setUniform2f(const char *name, float x, float y);

		void setUniform3f(const char *name, glm::vec3 value);

		void setUniform3f(const char *name, float x, float y, float z);

		void setUniform4f(const char *name, glm::vec4 value);

		void setUniform4f(const char *name, float x, float y, float z, float w);

		void setUniformMatrix4fv(const char *name, glm::mat4 value);

	private:
		// data members
		unsigned int m_Id;	 // shader program identifier
		unsigned int m_Vertex; // vertex shader identifier
		unsigned int m_Pixel;  // fragment shader identifier
	};
} // namespace gl