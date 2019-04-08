#include "Texture.h"
#include <iostream>

namespace gl
{
	// DEFAULT: 32-bit integer ARGB texture;
	// OPENCL:  128-bit floating point ARGB texture to be used with OpenCL.
	// ----------------------------------------------------------------------------
	Texture::Texture(unsigned int width, unsigned int height, unsigned int type) : m_Width(width), m_Height(height)
	{
		glGenTextures(1, &m_Id);
		glBindTexture(GL_TEXTURE_2D, m_Id);
		if (type == DEFAULT)
		{
			// regular texture
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, 0);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		}
		else if (type == SURFACE)
		{
			const auto byteCount = width * height * sizeof(core::Pixel);
			const auto *data = new unsigned int[byteCount];

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

			glGenBuffers(1, &m_PboId);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboId);
			glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, byteCount, data, GL_STREAM_DRAW_ARB);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
		else if (type == FLOAT_SURFACE)
		{
			const auto *data = new float[width * height * 4];

			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr);

			glGenBuffers(1, &m_PboId);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboId);
			glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * 4, data, GL_STREAM_DRAW_ARB);
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
		}
		else if (type == FLOAT)
		{
			// texture to be used with OpenCL code
			auto *data = new float[width * height * 4];
			memset(data, 0, width * height * 4 * sizeof(float));
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, data);
			glBindTexture(GL_TEXTURE_2D, m_Id);
		}
	}

	Texture::Texture(const char *fileName)
	{
		GLuint textureType = GL_TEXTURE_2D;
		glGenTextures(1, &m_Id);
		glBindTexture(textureType, m_Id);
		FREE_IMAGE_FORMAT fif;
		fif = FreeImage_GetFileType(fileName, 0);
		if (fif == FIF_UNKNOWN)
			fif = FreeImage_GetFIFFromFilename(fileName);
		FIBITMAP *tmp = FreeImage_Load(fif, fileName);
		if (!tmp)
			std::cout << "File not found: " << fileName << std::endl;
		FIBITMAP *dib = FreeImage_ConvertTo24Bits(tmp);
		FreeImage_Unload(tmp);
		unsigned int width = FreeImage_GetWidth(dib);
		unsigned int height = FreeImage_GetHeight(dib);
		auto *data = new unsigned int[width * height];
		auto *line = new unsigned char[width * 3];
		for (unsigned int y = 0; y < height; y++)
		{
			memcpy(line, FreeImage_GetScanLine(dib, height - 1 - y), width * 3);
			for (unsigned int x = 0; x < width; x++)
			{
				data[y * width + x] = (line[x * 3 + 2] << 16) + (line[x * 3 + 1] << 8) + line[x * 3 + 0];
			}
		}
		FreeImage_Unload(dib);
		glTexImage2D(textureType, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, data);
		delete[] data;
		glTexParameteri(textureType, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(textureType, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glGenerateMipmap(textureType);
	}

	void Texture::Bind() { glBindTexture(GL_TEXTURE_2D, m_Id); }

	void Texture::Bind(int slot)
	{
		glActiveTexture(slot);
		glBindTexture(GL_TEXTURE_2D, m_Id);
	}
	core::Pixel *Texture::mapToPixelBuffer()
	{
		glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
		Bind();
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboId);
		auto *buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
		memset(buffer, 0, m_Width * m_Height * 4);
		return (core::Pixel *)buffer;
	}

	core::FPixel *Texture::mapToFPixelBuffer()
	{
		glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
		Bind();
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboId);
		auto *buffer = glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
		memset(buffer, 0, m_Width * m_Height * 4);
		return (core::FPixel *)buffer;
	}

	void Texture::flushData()
	{
		glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
		Bind();
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboId);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Width, m_Height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
	}

	void Texture::clearBuffer()
	{
		glBindTexture(GL_TEXTURE_2D, m_Id);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboId);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Width, m_Height, GL_BGRA, GL_UNSIGNED_BYTE, 0);
	}
} // namespace gl