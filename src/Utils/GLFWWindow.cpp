#include "GLFWWindow.h"

#include "Timer.h"

namespace utils
{
	GLFWWindow *instance;

	GLFWWindow::GLFWWindow(const char *title, int width, int height, bool fullscreen, bool lockMouse)
		: Window(title, width, height), m_IsFullscreen(fullscreen)
	{
		instance = this;
		if (!glfwInit())
		{
			std::cout << "Could not init GLFW." << std::endl;
			exit(EXIT_FAILURE);
		}

		glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
		glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
		glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
		glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

		if (fullscreen)
		{
			m_Monitor = glfwGetPrimaryMonitor();
			const GLFWvidmode *mode = glfwGetVideoMode(m_Monitor);
			glfwWindowHint(GLFW_RED_BITS, mode->redBits);
			glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
			glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
			glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);
		}
		else
		{
			m_Monitor = nullptr;
		}

		m_Window = glfwCreateWindow(width, height, title, m_Monitor, nullptr);
		if (!m_Window)
		{
			std::cout << "Could not initialize GLFW Window." << std::endl;
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		glfwSetFramebufferSizeCallback(m_Window, FramebufferSizeCallback);
		glfwSetKeyCallback(m_Window, InputCallback);
		glfwSetErrorCallback(ErrorCallback);
		glfwSetCursorPosCallback(m_Window, MouseCallback);
		glfwSetScrollCallback(m_Window, MouseScrollCallback);
		glfwSetMouseButtonCallback(m_Window, MouseButtonCallback);
		if (lockMouse)
		{
			glfwSetInputMode(m_Window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
		}
		glfwMakeContextCurrent(m_Window);
		glfwShowWindow(m_Window);

		glewExperimental = GL_TRUE;
		GLenum error = glewInit();
		if (error != GLEW_OK)
		{
			std::cout << "Could not init GLEW: " << glewGetErrorString(error) << std::endl;
			exit(EXIT_FAILURE);
		}

		IMGUI_CHECKVERSION();
		m_ImGuiContext = ImGui::CreateContext();
		ImGui::StyleColorsDark();

		ImGui_ImplGlfw_InitForOpenGL(m_Window, true);
		ImGui_ImplOpenGL3_Init("#version 150");

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	GLFWWindow::~GLFWWindow()
	{
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		ImGui::DestroyContext(m_ImGuiContext);

		glfwDestroyWindow(m_Window);
		glfwTerminate();
	}

	void GLFWWindow::SetSize(int width, int height)
	{
		m_Width = width, m_Height = height;
		glfwSetWindowSize(m_Window, m_Width, m_Height);
	}

	void GLFWWindow::SetTitle(const char *title)
	{
		m_Title = title;
		glfwSetWindowTitle(m_Window, title);
	}

	void GLFWWindow::PollEvents() { glfwPollEvents(); }

	void GLFWWindow::Present()
	{
#if DEBUG
		GLenum error = glGetError();
		if (error != GL_NO_ERROR)
			std::cout << "OpenGL Error: " << error << std::endl;
#endif
		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(m_Window);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
	}

	int GLFWWindow::GetWidth()
	{
		glfwGetWindowSize(m_Window, &m_Width, &m_Height);
		return m_Width;
	}

	int GLFWWindow::GetHeight()
	{
		glfwGetWindowSize(m_Window, &m_Width, &m_Height);
		return m_Height;
	}

	void GLFWWindow::SetEventCallback(std::function<void(Event event)> callback) { m_OnEventCallback = callback; }

	void GLFWWindow::SetResizeCallback(std::function<void(int, int)> callback) { m_OnResizeCallback = callback; }

	void GLFWWindow::SwitchFullscreen()
	{
		static Timer t{};
		if (t.elapsed() < 1000.0f)
			return;

		t.reset();
		m_IsFullscreen = !m_IsFullscreen;

		if (m_IsFullscreen)
		{
			glfwGetWindowPos(m_Window, &m_WindowPos[0], &m_WindowPos[1]);
			glfwGetWindowSize(m_Window, &m_Width, &m_Height);

			m_Monitor = glfwGetPrimaryMonitor();
			const GLFWvidmode *mode = glfwGetVideoMode(m_Monitor);
			m_OnResizeCallback(mode->width, mode->height);

			glfwSetWindowMonitor(m_Window, m_Monitor, 0, 0, mode->width, mode->height, mode->refreshRate);
			glViewport(0, 0, mode->width, mode->height);
		}
		else
		{
			m_OnResizeCallback(m_Width, m_Height);
			glfwSetWindowMonitor(m_Window, nullptr, m_WindowPos[0], m_WindowPos[1], m_Width, m_Height, 0);
			glViewport(0, 0, m_Width, m_Height);
		}
	}

	void GLFWWindow::FramebufferSizeCallback(GLFWwindow *window, int width, int height)
	{
		instance->m_OnResizeCallback(width, height);
	}

	void GLFWWindow::InputCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
	{
		Event event;
		if (glfwWindowShouldClose(window))
			event.type = CLOSED;
		else
			event.type = KEY;

		event.key = key;
		if (action == GLFW_PRESS)
			event.state = KEY_PRESSED;
		else if (action == GLFW_RELEASE)
			event.state = KEY_RELEASED;

		instance->m_OnEventCallback(event);
	}

	void GLFWWindow::ErrorCallback(int code, const char *error)
	{
		std::cout << "ERROR (" << code << "): " << error << std::endl;
	}

	void GLFWWindow::MouseCallback(GLFWwindow *window, double xPos, double yPos)
	{
		Event event;
		event.type = MOUSE;
		event.state = MOUSE_MOVE;

		const float xPosition = float(xPos);
		const float yPosition = float(yPos);

		if (instance->m_FirstMouse)
		{
			instance->m_LastX = xPosition, instance->m_LastY = yPosition;
			instance->m_FirstMouse = false;
		}

		const float xOffset = xPosition - instance->m_LastX;
		const float yOffset = yPosition - instance->m_LastY;

		instance->m_LastX = xPosition;
		instance->m_LastY = yPosition;

		event.realX = xPos;
		event.realY = yPos;

		event.x = xOffset;
		event.y = yOffset;
		instance->m_OnEventCallback(event);
	}

	void GLFWWindow::MouseButtonCallback(GLFWwindow *window, int button, int action, int mods)
	{
		Event event;
		event.type = MOUSE;
		event.state = action == GLFW_PRESS ? KEY_PRESSED : KEY_RELEASED;
		event.key = button;

		instance->m_OnEventCallback(event);
	}

	void GLFWWindow::MouseScrollCallback(GLFWwindow *window, double xOffset, double yOffset)
	{
		Event event;
		event.type = MOUSE;
		event.state = MOUSE_SCROLL;

		event.x = float(xOffset);
		event.y = float(yOffset);

		instance->m_OnEventCallback(event);
	}

	void GLFWWindow::Clear(const glm::vec4 &color) { glClearColor(color.x, color.y, color.z, color.w); }
} // namespace utils