#pragma once

#include <glm/glm.hpp>

#include <Tracer/Core/Ray.cuh>

namespace core
{
#define ROTATION_SPEED 0.001f
#define MOVEMENT_SPEED 0.005f

	class Camera
	{
	public:
		Camera() = default;

		Camera(int width, int height, float fov, float mouseSens = ROTATION_SPEED,
			glm::vec3 pos = glm::vec3(0.f, 0.f, 1.f));

		Ray generateRay(float x, float y) const;

		Ray generateRandomRay(float x, float y, float r1, float r2) const;

		bool processMouse(int x, int y) noexcept;

		bool processMouse(float x, float y) noexcept;

		glm::vec3 getPosition() const noexcept;

		void move(vec3 offset) noexcept;

		void rotate(vec2 offset) noexcept;

		bool changeFov(float fov) noexcept;

		const float& getFov() const noexcept;

		const float& getPitch() const noexcept;

		const float& getYaw() const noexcept;

		inline float getPlaneDistance() const noexcept { return m_PlaneDistance; }

		inline float getInvWidth() const noexcept { return m_InvWidth; }

		inline float getInvHeight() const noexcept { return m_InvHeight; }

		inline float getAspectRatio() const noexcept { return m_AspectRatio; }

		inline glm::vec3 getUp() const noexcept { return m_Up; }

		inline void setPosition(glm::vec3 position) noexcept { this->m_Position = position; }

		inline glm::vec3 getForward() const noexcept { return m_Forward; }

		inline void setWidth(int width)
		{
			m_Width = float(width);
			m_InvWidth = (float)1.0f / float(width);
			m_AspectRatio = float(m_Width) / float(m_Height);
		}

		inline void setHeight(int height)
		{
			m_Height = float(height);
			m_InvHeight = (float)1.0f / float(height);
			m_AspectRatio = float(m_Width) / float(m_Height);
		}

		inline void setDimensions(int width, int height)
		{
			m_Width = float(width);
			m_Height = float(height);
			m_InvWidth = (float)1.0f / float(width);
			m_InvHeight = (float)1.0f / float(height);
			m_AspectRatio = float(width) / float(height);
		}

		bool handleKeys(const std::vector<bool>& keys, float speed = 1.0f);
	public:
		glm::vec3 m_Up = glm::vec3(0.0f, 1.0f, 0.0f);
		glm::vec3 m_Forward;
		glm::vec3 m_Position;
		float m_Fov;

		float m_PlaneDistance;
		float m_RotationSpeed = 0.005f;
		float m_Width, m_Height;
		float m_InvWidth, m_InvHeight;
		float m_AspectRatio{};
		float m_Yaw = -0.04f;
		float m_Pitch = 0.2f;
	};
} // namespace core
