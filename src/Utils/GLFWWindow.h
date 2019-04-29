#include "Window.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <functional>

#include "../ImGui/ImGuiGLFW.h"
#include "../ImGui/ImGuiOpenGL3.h"

namespace utils
{
	class GLFWWindow : public Window
	{
	public:
		GLFWWindow(const char *title, int width, int height, bool fullscreen = false, bool lockMouse = false);
		~GLFWWindow();

		void setSize(int width, int height) override;
		void setTitle(const char *title) override;
		void pollEvents() override;
		void clear(const glm::vec4 &color) override;
		void present() override;
		inline GLFWwindow *getWindow()
		{
			return m_Window;
		}

		int getWidth() override;
		int getHeight() override;

		void switchFullscreen() override;
		bool shouldClose() override;
		void close() override;
		void setVsync(bool status) override;

		static void FramebufferSizeCallback(GLFWwindow *window, int width, int height);
		static void InputCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
		static void ErrorCallback(int code, const char *error);
		static void MouseCallback(GLFWwindow *window, double xPos, double yPos);
		static void MouseButtonCallback(GLFWwindow *window, int button, int action, int mods);
		static void MouseScrollCallback(GLFWwindow *window, double xOffset, double yOffset);

		GLFWmonitor *m_Monitor = nullptr;

		float m_LastX = 0, m_LastY = 0;
		int m_WindowPos[2] = { 0, 0 };
		int m_WindowSize[2] = { 0 ,0 };
		bool m_FirstMouse = true;
		bool m_IsFullscreen = false;

	private:
		GLFWwindow *m_Window;
	};
} // namespace utils