#pragma once

#include "Utils/Surface.h"
#include <GL/glew.h>

namespace gl
{
	class Texture
	{
	public:
		enum
		{
			DEFAULT = 0,
			FLOAT = 1,
			SURFACE = 2,
			FLOAT_SURFACE = 3
		};

		// constructor / destructor
		Texture(unsigned int width, unsigned int height, unsigned int type = DEFAULT);

		Texture(const char *fileName);

		// get / set
		unsigned int GetID() { return m_Id; }

		// methods
		void Bind();

		void Bind(int slot);

		core::Pixel *mapToPixelBuffer();

		core::FPixel *mapToFPixelBuffer();

		void clearBuffer();

		void flushData();

		inline int GetWidth() { return m_Width; }

		inline int GetHeight() { return m_Height; }

	public:
		// data members
		GLuint m_Id;
		GLuint m_PboId = 0;
		int m_Width, m_Height;
	};
} // namespace gl
