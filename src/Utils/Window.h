#pragma once

#include <glm/glm.hpp>
#include <iostream>
#include <functional>

#include "ImGui/imgui.h"

namespace utils
{
	class Window
	{
	public:
		enum EventType
		{
			KEY = 0,
			MOUSE = 1,
			CLOSED = 2
		};

		enum State
		{
			KEY_PRESSED = 0,
			KEY_RELEASED = 1,
			MOUSE_SCROLL = 2,
			MOUSE_MOVE = 3,
		};

		struct Event
		{
			EventType type;
			int key;
			State state;
			float x, y;
			double realX, realY;
		};

		Window(const char* title, int width, int height) : m_Title(title), m_Width(width), m_Height(height) {}
		virtual ~Window() = default;

		virtual int getWidth() { return m_Width; }
		virtual int getHeight() { return m_Height; }

		virtual void setSize(int width, int height) = 0;
		virtual void setTitle(const char* title) = 0;
		virtual void pollEvents() = 0;
		virtual void clear(const glm::vec4& color) = 0;
		virtual void present() = 0;
		virtual void switchFullscreen() = 0;
		virtual bool shouldClose() = 0;
		virtual void close() = 0;
		virtual void setVsync(bool status) = 0;

		void setEventCallback(std::function<void(Window::Event event)> callback);
		void setResizeCallback(std::function<void(int, int)> callback);

	protected:
		const char* m_Title;
		int m_Width, m_Height;

		std::function<void(Event event)> m_OnEventCallback = [](Event) {};
		std::function<void(int, int)> m_OnResizeCallback = [](int, int) {};

		ImGuiContext* m_ImGuiContext = nullptr;
	};

	inline void Window::setEventCallback(std::function<void(Event event)> callback)
	{
		m_OnEventCallback = callback;
	}

	inline void Window::setResizeCallback(std::function<void(int, int)> callback)
	{
		m_OnResizeCallback = callback;
	}


} // namespace utils