#include "GL/GLTools.h"
#include "GL/Shader.h"

#include <glm/ext.hpp>

using namespace std;
using namespace glm;

namespace gl
{
	inline string textFileRead(const char* _File)
	{
		string data, line;
		ifstream f(_File);
		if (f.is_open())
		{
			while (!f.eof())
			{
				getline(f, line);
				data.append(line);
				data.append("\n");
			}
			f.close();
		}
		return data;
	}

	Shader::Shader(const char* vfile, const char* pfile) { Init(vfile, pfile); }

	Shader::~Shader()
	{
		glDetachShader(m_Id, m_Pixel);
		glDetachShader(m_Id, m_Vertex);
		glDeleteShader(m_Pixel);
		glDeleteShader(m_Vertex);
		glDeleteProgram(m_Id);
	}

	void Shader::Init(const char* vfile, const char* pfile)
	{
		string vsText = textFileRead(vfile);
		string fsText = textFileRead(pfile);

		if (vsText.empty())
			std::cout << "File not found" << vfile << std::endl, exit(1);

		if (fsText.empty())
			std::cout << "File not found" << pfile << std::endl, exit(1);

		const char* vertexText = vsText.c_str();
		const char* fragmentText = fsText.c_str();
		Compile(vertexText, fragmentText);
	}

	void Shader::Compile(const char* vtext, const char* ftext)
	{
		m_Vertex = glCreateShader(GL_VERTEX_SHADER);
		m_Pixel = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(m_Vertex, 1, &vtext, 0);
		glCompileShader(m_Vertex);
		CheckShader(m_Vertex);
		glShaderSource(m_Pixel, 1, &ftext, 0);
		glCompileShader(m_Pixel);
		CheckShader(m_Pixel);
		m_Id = glCreateProgram();
		glAttachShader(m_Id, m_Vertex);
		glAttachShader(m_Id, m_Pixel);
		glLinkProgram(m_Id);
		CheckProgram(m_Id);
	}

	void Shader::Bind() { glUseProgram(m_Id); }

	void Shader::SetInputTexture(unsigned int slot, const char* name, Texture* texture)
	{
		glActiveTexture(slot);
		glBindTexture(GL_TEXTURE_2D, texture->GetID());
		glUniform1i(getUniformLocation(name), slot);
	}

	void Shader::SetInputMatrix(const char* name, const mat4& matrix)
	{
		glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, (const GLfloat*)&matrix);
	}

	void Shader::Unbind() { glUseProgram(0); }

	GLint Shader::getUniformLocation(const char* name) { return glGetUniformLocation(this->m_Id, name); }

	void Shader::setUniform1f(const char* name, float value) { glUniform1f(getUniformLocation(name), value); }
	void Shader::setUniform1i(const char* name, int value) { glUniform1i(getUniformLocation(name), value); }

	void Shader::setUniform2f(const char* name, vec2 value) { glUniform2f(getUniformLocation(name), value.x, value.y); }
	void Shader::setUniform2f(const char* name, float x, float y) { glUniform2f(getUniformLocation(name), x, y); }
	void Shader::setUniform3f(const char* name, vec3 value)
	{
		glUniform3f(getUniformLocation(name), value.x, value.y, value.z);
	}
	void Shader::setUniform3f(const char* name, float x, float y, float z)
	{
		glUniform3f(getUniformLocation(name), x, y, z);
	}
	void Shader::setUniform4f(const char* name, vec4 value)
	{
		glUniform4f(getUniformLocation(name), value.x, value.y, value.z, value.w);
	}
	void Shader::setUniform4f(const char* name, float x, float y, float z, float w)
	{
		glUniform4f(getUniformLocation(name), x, y, z, w);
	}
	void Shader::setUniformMatrix4fv(const char* name, mat4 value)
	{
		glUniformMatrix4fv(getUniformLocation(name), 1, GL_FALSE, (const GLfloat*)&value);
	}
} // namespace gl