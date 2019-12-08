#include "Utils/Surface.h"
#include <iostream>
#include <fstream>

namespace core
{
	Surface::Surface(int width, int height, Pixel* buffer, int pitch)
		: m_Buffer(buffer), m_Width(width), m_Height(height), m_Pitch(pitch)
	{
		m_Flags = 0;
	}

	Surface::Surface(int width, int height) : m_Width(width), m_Height(height), m_Pitch(width)
	{
		m_Buffer = new Pixel[size_t(width * height)];
		m_Flags = OWNER;
	}

	Surface::Surface(const char* file) : m_Buffer(nullptr), m_Width(0), m_Height(0)
	{
		std::ifstream f(file);
		if (!f.is_open())
		{
			std::string t = std::string("File not found: ") + file;
			const char* msg = t.c_str();
			std::cout << msg << std::endl;
			throw std::runtime_error(msg);
		}

		LoadImage(file);
	}

	void Surface::LoadImage(const char* file)
	{
		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
		fif = FreeImage_GetFileType(file, 0);
		if (fif == FIF_UNKNOWN)
			fif = FreeImage_GetFIFFromFilename(file);

		if (fif == FIF_HDR)
		{
			FIBITMAP* tmp = FreeImage_Load(fif, file);
			FIBITMAP* dib = FreeImage_ConvertToRGBAF(tmp);

			m_Width = m_Pitch = FreeImage_GetWidth(dib);
			m_Height = FreeImage_GetHeight(dib);
			m_Buffer = new Pixel[m_Width * m_Height];

			m_TexBuffer.clear();
			m_TexBuffer.reserve(m_Width * m_Height);
			m_Flags = OWNER;

			for (int y = 0; y < m_Height; y++)
			{
				const FIRGBAF* bits = (FIRGBAF*)FreeImage_GetScanLine(dib, y);
				for (int x = 0; x < m_Width; x++)
				{
					const glm::vec4 color = { bits[x].red, bits[x].green, bits[x].blue, bits[x].alpha };
					const unsigned int red = glm::clamp(bits[x].red, 0.0f, 1.0f) * 255;
					const unsigned int green = glm::clamp(bits[x].green, 0.0f, 1.0f) * 255;
					const unsigned int blue = glm::clamp(bits[x].blue, 0.0f, 1.0f) * 255;
					const unsigned int alpha = glm::clamp(bits[x].alpha, 0.0f, 1.0f) * 255;

					m_Buffer[x + y * m_Width] = (red << 0) | (green << 8) | (blue << 16) | (alpha << 24);
					m_TexBuffer.push_back(color);
				}
			}

			FreeImage_Unload(dib);
		}
		else
		{
			FIBITMAP* tmp = FreeImage_Load(fif, file);
			FIBITMAP* dib = FreeImage_ConvertTo32Bits(tmp);
			FreeImage_Unload(tmp);

			m_Width = m_Pitch = FreeImage_GetWidth(dib);
			m_Height = FreeImage_GetHeight(dib);
			m_Buffer = new Pixel[m_Width * m_Height];

			m_TexBuffer.clear();
			m_TexBuffer.reserve(m_Width * m_Height);
			m_Flags = OWNER;

			for (int y = 0; y < m_Height; y++)
			{
				for (int x = 0; x < m_Width; x++)
				{
					RGBQUAD quad;
					FreeImage_GetPixelColor(dib, x, y, &quad);
					const unsigned int red = (unsigned int)(quad.rgbRed);
					const unsigned int green = (unsigned int)(quad.rgbGreen);
					const unsigned int blue = (unsigned int)(quad.rgbBlue);
					m_Buffer[x + y * m_Width] = (red << 0) | (green << 8) | (blue << 16);
					m_TexBuffer.emplace_back(float(red) / 255.0f, float(green) / 255.0f, float(blue) / 255.0f, 1.0f);
				}
			}
			FreeImage_Unload(dib);
		}
	}

	Surface::~Surface()
	{
		if (m_Flags & OWNER)
		{
			delete[] m_Buffer;
		}
	}

	void Surface::SetBuffer(Pixel * buffer)
	{
		if (m_Flags == OWNER)
		{
			delete[] m_Buffer;
			m_Flags = 0;
		}

		m_Buffer = buffer;
	}

	void Surface::Clear(Pixel color)
	{
		int s = m_Width * m_Height;
		for (int i = 0; i < s; i++)
			m_Buffer[i] = color;
	}

	void Surface::Resize(Surface * original)
	{
		Pixel* src = original->GetBuffer(), * dst = m_Buffer;
		int u, v, owidth = original->GetWidth(), oheight = original->GetHeight();
		int dx = (owidth << 10) / m_Width, dy = (oheight << 10) / m_Height;
		for (v = 0; v < m_Height; v++)
		{
			for (u = 0; u < m_Width; u++)
			{
				int su = u * dx, sv = v * dy;
				Pixel* s = src + (su >> 10) + (sv >> 10) * owidth;
				int ufrac = su & 1023, vfrac = sv & 1023;
				int w4 = (ufrac * vfrac) >> 12;
				int w3 = ((1023 - ufrac) * vfrac) >> 12;
				int w2 = (ufrac * (1023 - vfrac)) >> 12;
				int w1 = ((1023 - ufrac) * (1023 - vfrac)) >> 12;
				int x2 = ((su + dx) > ((owidth - 1) << 10)) ? 0 : 1;
				int y2 = ((sv + dy) > ((oheight - 1) << 10)) ? 0 : 1;
				Pixel p1 = *s, p2 = *(s + x2), p3 = *(s + owidth * y2), p4 = *(s + owidth * y2 + x2);
				unsigned int r =
					(((p1 & REDMASK) * w1 + (p2 & REDMASK) * w2 + (p3 & REDMASK) * w3 + (p4 & REDMASK) * w4) >> 8) &
					REDMASK;
				unsigned int g =
					(((p1 & GREENMASK) * w1 + (p2 & GREENMASK) * w2 + (p3 & GREENMASK) * w3 + (p4 & GREENMASK) * w4) >> 8) &
					GREENMASK;
				unsigned int b =
					(((p1 & BLUEMASK) * w1 + (p2 & BLUEMASK) * w2 + (p3 & BLUEMASK) * w3 + (p4 & BLUEMASK) * w4) >> 8) &
					BLUEMASK;
				*(dst + u + v * m_Pitch) = (Pixel)(r + g + b);
			}
		}
	}

	const glm::vec3 Surface::GetColorAt(const glm::vec2 texCoords) const
	{
		const int yValue = (int)glm::max(0, (int)(glm::min(texCoords.y, 1.f) * m_Height) - 1);
		const int xValue = (int)glm::max(0, (int)(glm::min(texCoords.x, 1.f) * m_Width) - 1);

		return m_TexBuffer[xValue + yValue * m_Pitch];
	}

	const glm::vec3 Surface::GetColorAt(const float& x, const float& y) const
	{
		const int yValue = (int)glm::max(0, (int)(glm::min(y, 1.f) * m_Height) - 1);
		const int xValue = (int)glm::max(0, (int)(glm::min(x, 1.f) * m_Width) - 1);

		return m_TexBuffer[xValue + yValue * m_Pitch];
	}

#define OUTCODE(x, y) (((x) < xmin) ? 1 : (((x) > xmax) ? 2 : 0)) + (((y) < ymin) ? 4 : (((y) > ymax) ? 8 : 0))

	void Surface::Line(float x1, float y1, float x2, float y2, Pixel c)
	{
		// clip (Cohen-Sutherland,
		// https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm)
		const float xmin = 0, ymin = 0, xmax = (float)m_Width - 1, ymax = (float)m_Height - 1;
		int c0 = OUTCODE(x1, y1), c1 = OUTCODE(x2, y2);
		bool accept = false;
		while (1)
		{
			if (!(c0 | c1))
			{
				accept = true;
				break;
			}
			else if (c0 & c1)
				break;
			else
			{
				float x, y;
				const int co = c0 ? c0 : c1;
				if (co & 8)
					x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1), y = ymax;
				else if (co & 4)
					x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1), y = ymin;
				else if (co & 2)
					y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1), x = xmax;
				else if (co & 1)
					y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1), x = xmin;
				if (co == c0)
					x1 = x, y1 = y, c0 = OUTCODE(x1, y1);
				else
					x2 = x, y2 = y, c1 = OUTCODE(x2, y2);
			}
		}
		if (!accept)
			return;
		float b = x2 - x1;
		float h = y2 - y1;
		float l = fabsf(b);
		if (fabsf(h) > l)
			l = fabsf(h);
		int il = (int)l;
		float dx = b / (float)l;
		float dy = h / (float)l;
		for (int i = 0; i <= il; i++)
		{
			*(m_Buffer + (int)x1 + (int)y1 * m_Pitch) = c;
			x1 += dx, y1 += dy;
		}
	}

	void Surface::Plot(int x, int y, Pixel c) { m_Buffer[x + y * m_Pitch] = c; }

	void Surface::Plot(int x, int y, const glm::vec3 & color)
	{
		glm::vec3 col = glm::sqrt(glm::clamp(color, 0.0f, 1.0f)) * 255.99f;
		const unsigned int red = (unsigned int)(col.r);
		const unsigned int green = (unsigned int)(col.g) << 8;
		const unsigned int blue = (unsigned int)(col.b) << 16;
		Plot(x, y, red + green + blue);
	}

	void Surface::Box(int x1, int y1, int x2, int y2, Pixel c)
	{
		Line((float)x1, (float)y1, (float)x2, (float)y1, c);
		Line((float)x2, (float)y1, (float)x2, (float)y2, c);
		Line((float)x1, (float)y2, (float)x2, (float)y2, c);
		Line((float)x1, (float)y1, (float)x1, (float)y2, c);
	}

	void Surface::Bar(int x1, int y1, int x2, int y2, Pixel c)
	{
		Pixel* a = x1 + y1 * m_Pitch + m_Buffer;
		for (int y = y1; y <= y2; y++)
		{
			for (int x = 0; x <= (x2 - x1); x++)
				a[x] = c;
			a += m_Pitch;
		}
	}

	void Surface::CopyTo(Surface * destination, int a_X, int a_Y)
	{
		Pixel* dst = destination->GetBuffer();
		Pixel* src = m_Buffer;
		if ((src) && (dst))
		{
			int srcwidth = m_Width;
			int srcheight = m_Height;
			int srcpitch = m_Pitch;
			int dstwidth = destination->GetWidth();
			int dstheight = destination->GetHeight();
			int dstpitch = destination->GetPitch();
			if ((srcwidth + a_X) > dstwidth)
				srcwidth = dstwidth - a_X;
			if ((srcheight + a_Y) > dstheight)
				srcheight = dstheight - a_Y;
			if (a_X < 0)
				src -= a_X, srcwidth += a_X, a_X = 0;
			if (a_Y < 0)
				src -= a_Y * srcpitch, srcheight += a_Y, a_Y = 0;
			if ((srcwidth > 0) && (srcheight > 0))
			{
				dst += a_X + dstpitch * a_Y;
				for (int y = 0; y < srcheight; y++)
				{
					memcpy(dst, src, srcwidth * 4);
					dst += dstpitch;
					src += srcpitch;
				}
			}
		}
	}

	void Surface::BlendCopyTo(Surface * destination, int X, int Y)
	{
		Pixel* dst = destination->GetBuffer();
		Pixel* src = m_Buffer;
		if ((src) && (dst))
		{
			int srcwidth = m_Width;
			int srcheight = m_Height;
			int srcpitch = m_Pitch;
			int dstwidth = destination->GetWidth();
			int dstheight = destination->GetHeight();
			int dstpitch = destination->GetPitch();
			if ((srcwidth + X) > dstwidth)
				srcwidth = dstwidth - X;
			if ((srcheight + Y) > dstheight)
				srcheight = dstheight - Y;
			if (X < 0)
				src -= X, srcwidth += X, X = 0;
			if (Y < 0)
				src -= Y * srcpitch, srcheight += Y, Y = 0;
			if ((srcwidth > 0) && (srcheight > 0))
			{
				dst += X + dstpitch * Y;
				for (int y = 0; y < srcheight; y++)
				{
					for (int x = 0; x < srcwidth; x++)
						dst[x] = AddBlend(dst[x], src[x]);
					dst += dstpitch;
					src += srcpitch;
				}
			}
		}
	}

	void Surface::ScaleColor(unsigned int scale)
	{
		int s = m_Pitch * m_Height;
		for (int i = 0; i < s; i++)
		{
			Pixel c = m_Buffer[i];
			unsigned int rb = (((c & (REDMASK | BLUEMASK)) * scale) >> 5) & (REDMASK | BLUEMASK);
			unsigned int g = (((c & GREENMASK) * scale) >> 5) & GREENMASK;
			m_Buffer[i] = rb + g;
		}
	}

	glm::vec4 * Surface::GetTextureBuffer() { return m_TexBuffer.data(); }

	Sprite::Sprite(Surface * output, unsigned int numFrames)
		: m_Width(output->GetWidth() / numFrames), m_Height(output->GetHeight()), m_Pitch(output->GetWidth()),
		m_NumFrames(numFrames), m_CurrentFrame(0), m_Flags(0), m_Start(new unsigned int* [numFrames]), m_Surface(output)
	{
		InitializeStartData();
	}

	Sprite::~Sprite()
	{
		delete m_Surface;
		for (unsigned int i = 0; i < m_NumFrames; i++)
			delete m_Start[i];
		delete[] m_Start;
	}

	void Sprite::Draw(Surface * output, int X, int Y)
	{
		if ((X < -m_Width) || (X > (output->GetWidth() + m_Width)))
			return;
		if ((Y < -m_Height) || (Y > (output->GetHeight() + m_Height)))
			return;
		int x1 = X, x2 = X + m_Width;
		int y1 = Y, y2 = Y + m_Height;
		Pixel * src = GetBuffer() + m_CurrentFrame * m_Width;
		if (x1 < 0)
		{
			src += -x1;
			x1 = 0;
		}
		if (x2 > output->GetWidth())
			x2 = output->GetWidth();
		if (y1 < 0)
		{
			src += -y1 * m_Pitch;
			y1 = 0;
		}
		if (y2 > output->GetHeight())
			y2 = output->GetHeight();
		Pixel* dest = output->GetBuffer();
		int xs;
		const int dpitch = output->GetPitch();
		if ((x2 > x1) && (y2 > y1))
		{
			unsigned int addr = y1 * dpitch + x1;
			const int width = x2 - x1;
			const int height = y2 - y1;
			for (int y = 0; y < height; y++)
			{
				const int line = y + (y1 - Y);
				const int lsx = m_Start[m_CurrentFrame][line] + X;
				if (m_Flags & FLARE)
				{
					xs = (lsx > x1) ? lsx - x1 : 0;
					for (int x = xs; x < width; x++)
					{
						const Pixel c1 = *(src + x);
						if (c1 & 0xffffff)
						{
							const Pixel c2 = *(dest + addr + x);
							*(dest + addr + x) = AddBlend(c1, c2);
						}
					}
				}
				else
				{
					xs = (lsx > x1) ? lsx - x1 : 0;
					for (int x = xs; x < width; x++)
					{
						const Pixel c1 = *(src + x);
						if (c1 & 0xffffff)
							* (dest + addr + x) = c1;
					}
				}
				addr += dpitch;
				src += m_Pitch;
			}
		}
	}

	void Sprite::DrawScaled(int X, int Y, int width, int height, Surface * output)
	{
		if ((width == 0) || (height == 0))
			return;
		for (int x = 0; x < width; x++)
			for (int y = 0; y < height; y++)
			{
				int u = (int)((float)x * ((float)m_Width / (float)width));
				int v = (int)((float)y * ((float)m_Height / (float)height));
				Pixel color = GetBuffer()[u + v * m_Pitch];
				if (color & 0xffffff)
					output->GetBuffer()[X + x + ((Y + y) * output->GetPitch())] = color;
			}
	}

	void Sprite::InitializeStartData()
	{
		for (unsigned int f = 0; f < m_NumFrames; ++f)
		{
			m_Start[f] = new unsigned int[m_Height];
			for (int y = 0; y < m_Height; ++y)
			{
				m_Start[f][y] = m_Width;
				Pixel* addr = GetBuffer() + f * m_Width + y * m_Pitch;
				for (int x = 0; x < m_Width; ++x)
				{
					if (addr[x])
					{
						m_Start[f][y] = x;
						break;
					}
				}
			}
		}
	}
} // namespace core