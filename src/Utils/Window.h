#pragma once

#include <glm/glm.hpp>
#include <iostream>

namespace utils
{
	class Window
	{
	public:
		Window(const char* title, int width, int height) : m_Title(title), m_Width(width), m_Height(height) {}
		virtual ~Window() = default;

		virtual int GetWidth() { return m_Width; }
		virtual int GetHeight() { return m_Height; }

		virtual void SetSize(int width, int height) = 0;
		virtual void SetTitle(const char* title) = 0;
		virtual void PollEvents() = 0;
		virtual void Clear(const glm::vec4& color) = 0;
		virtual void Present() = 0;
		virtual void SwitchFullscreen() = 0;

	protected:
		const char* m_Title;
		int m_Width, m_Height;
	};
} // namespace utils


//#ifdef WIN32
////#define WIN32_LEAN_AND_MEAN 1
//#include <Windows.h>
//
//int main(int argc, char* argv[]);
//
//int __stdcall WinMain(HANDLE hInstance, HANDLE hPrevInstance, char* lpCmdLine, int nShowCmd)
//{
//	return main(__argc, __argv);
//}
//
//#endif