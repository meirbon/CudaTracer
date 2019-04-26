#include "Camera.h"

using namespace glm;

namespace core
{
	Camera::Camera(int width, int height, float fov, float mouseSens, glm::vec3 pos)
		: m_Width((float)width), m_Height((float)height), m_InvWidth(1.f / width), m_InvHeight(1.f / height)
	{
		m_RotationSpeed = mouseSens;
		m_ViewDirection = getDirectionFromPitchAndYaw();
		m_Position = pos;
		m_AspectRatio = m_Width / m_Height;
		m_FOV_Distance = tanf(glm::radians(fov * 0.5f));
	}

	Ray Camera::GenerateRay(float x, float y) const
	{
		const vec3 &w = m_ViewDirection;
		const vec3 u = normalize(cross(w, m_Up));
		const vec3 v = normalize(cross(u, w));

		const vec3 horizontal = u * m_FOV_Distance * m_AspectRatio;
		const vec3 vertical = v * m_FOV_Distance;

		const float PixelX = x * m_InvWidth;
		const float PixelY = y * m_InvHeight;

		const float ScreenX = 2.f * PixelX - 1.f;
		const float ScreenY = 1.f - 2.f * PixelY;

		const vec3 pointAtDistanceOneFromPlane = w + horizontal * ScreenX + vertical * ScreenY;

		return { m_Position, normalize(pointAtDistanceOneFromPlane), 0 };
	}

	Ray Camera::GenerateRandomRay(float x, float y, float r1, float r2) const
	{
		const float newX = x + r1 - .5f;
		const float newY = y + r2 - .5f;

		return GenerateRay(newX, newY);
	}

	void Camera::ProcessMouse(int x, int y) noexcept
	{
		RotateRight((float)x);
		RotateUp((float)y);
		m_ViewDirection = getDirectionFromPitchAndYaw();
	}

	void Camera::ProcessMouse(float x, float y) noexcept
	{
		RotateRight(x);
		RotateUp(y);
		m_ViewDirection = getDirectionFromPitchAndYaw();
	}

	vec3 Camera::GetPosition() const noexcept { return m_Position; }

	void Camera::MoveUp(float movementSpeed) noexcept { m_Position += m_Up * movementSpeed; }

	void Camera::MoveDown(float movementSpeed) noexcept { m_Position -= m_Up * movementSpeed; }

	void Camera::MoveForward(float movementSpeed) noexcept { m_Position += m_ViewDirection * (movementSpeed); }

	void Camera::MoveBackward(float movementSpeed) noexcept { m_Position += m_ViewDirection * (-movementSpeed); }

	void Camera::MoveLeft(float movementSpeed) noexcept
	{
		const vec3 axis = cross(m_ViewDirection, m_Up);
		m_Position += normalize(axis) * (-movementSpeed);
	}

	void Camera::MoveRight(float movementSpeed) noexcept
	{
		const vec3 axis = cross(m_ViewDirection, m_Up);
		m_Position += normalize(axis) * movementSpeed;
	}

	void Camera::RotateUp(float times) noexcept
	{
		m_Pitch = glm::clamp(m_Pitch + times * m_RotationSpeed, glm::radians(-89.0f), glm::radians(89.0f));
		m_ViewDirection = getDirectionFromPitchAndYaw();
	}

	void Camera::RotateDown(float times) noexcept
	{
		m_Pitch = glm::clamp(m_Pitch - times * m_RotationSpeed, glm::radians(-89.0f), glm::radians(89.0f));
		m_ViewDirection = getDirectionFromPitchAndYaw();
	}

	void Camera::RotateRight(float times) noexcept
	{
		m_Yaw += times * m_RotationSpeed;
		m_ViewDirection = getDirectionFromPitchAndYaw();
	}

	void Camera::RotateLeft(float times) noexcept
	{
		m_Yaw -= times * m_RotationSpeed;
		m_ViewDirection = getDirectionFromPitchAndYaw();
	}

	void Camera::ChangeFOV(float fov) noexcept
	{
		if (fov < 20.f)
			fov = 20.f;
		else if (fov > 160.f)
			fov = 160.f;
		m_FOV.x = m_FOV.y = fov;
		m_FOV_Distance = tanf(glm::radians(fov * 0.5f));
	}

	const float &Camera::GetFOV() const noexcept { return m_FOV.x; }

	vec3 Camera::getDirectionFromPitchAndYaw() noexcept
	{
		return vec3(sinf(m_Yaw) * cosf(m_Pitch), sinf(m_Pitch), -1.0f * cosf(m_Yaw) * cosf(m_Pitch));
	}

	const float &Camera::GetPitch() const noexcept { return m_Pitch; }

	const float &Camera::GetYaw() const noexcept { return m_Yaw; }
} // namespace core