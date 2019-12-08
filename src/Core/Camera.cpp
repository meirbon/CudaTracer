#include "Core/Camera.h"

#include <GLFW/glfw3.h>

using namespace glm;

namespace core
{
	Camera::Camera(int width, int height, float fov, float mouseSens, glm::vec3 pos)
		: m_Width((float)width), m_Height((float)height), m_InvWidth(1.f / width), m_InvHeight(1.f / height)
	{
		m_RotationSpeed = mouseSens;
		m_Forward = vec3(0, 0, -1);
		m_Position = pos;
		m_AspectRatio = m_Width / m_Height;
		m_PlaneDistance = tanf(glm::radians(fov * 0.5f));
	}

	Ray Camera::generateRay(float x, float y) const
	{
		const vec3& w = m_Forward;
		const vec3 u = normalize(cross(w, m_Up));
		const vec3 v = normalize(cross(u, w));

		const vec3 horizontal = u * m_PlaneDistance * m_AspectRatio;
		const vec3 vertical = v * m_PlaneDistance;

		const float PixelX = x * m_InvWidth;
		const float PixelY = y * m_InvHeight;

		const float ScreenX = 2.f * PixelX - 1.f;
		const float ScreenY = 1.f - 2.f * PixelY;

		const vec3 pointAtDistanceOneFromPlane = w + horizontal * ScreenX + vertical * ScreenY;

		return { m_Position, normalize(pointAtDistanceOneFromPlane), 0 };
	}

	Ray Camera::generateRandomRay(float x, float y, float r1, float r2) const
	{
		const float newX = x + r1 - .5f;
		const float newY = y + r2 - .5f;

		return generateRay(newX, newY);
	}

	bool Camera::processMouse(int x, int y) noexcept
	{
		if (x != 0 || y != 0)
		{
			rotate({ float(x), float(y) });
			return true;
		}

		return false;
	}

	bool Camera::processMouse(float x, float y) noexcept
	{
		if (x != 0.0f || y != 0.0f)
		{
			rotate({ x, y });
			return true;
		}

		return false;
	}

	vec3 Camera::getPosition() const noexcept { return m_Position; }

	void Camera::move(vec3 offset) noexcept
	{
		vec3 tmp = m_Up;
		if (m_Forward.y > 0.99f)
			tmp = vec3(0, 0, 1);

		const vec3 right = cross(m_Forward, tmp);

		m_Position += offset.x * right + offset.y * m_Up + offset.z * m_Forward;
	}

	void Camera::rotate(vec2 offset) noexcept
	{
		m_Pitch = glm::clamp(m_Pitch + offset.y * m_RotationSpeed, glm::radians(-89.0f), glm::radians(89.0f));
		m_Yaw += offset.x * m_RotationSpeed;
		m_Forward = vec3(sinf(m_Yaw) * cosf(m_Pitch), sinf(m_Pitch), -1.0f * cosf(m_Yaw) * cosf(m_Pitch));
	}

	bool Camera::changeFov(float fov) noexcept
	{
		if (fov != 0)
		{
			m_Fov = clamp(m_Fov + fov, 20.0f, 160.0f);
			m_PlaneDistance = tanf(glm::radians(m_Fov * 0.5f));
			return true;
		}
		return false;
	}

	const float& Camera::getFov() const noexcept { return m_Fov; }

	const float& Camera::getPitch() const noexcept { return m_Pitch; }

	const float& Camera::getYaw() const noexcept { return m_Yaw; }

	bool Camera::handleKeys(const std::vector<bool> & keys, float speed)
	{
		if (keys[GLFW_KEY_LEFT_SHIFT])
			speed *= 3.0f;

		vec3 offset = vec3(0.0f);
		vec2 viewOffset = vec2(0.0f);

		if (keys[GLFW_KEY_A])
			offset.x -= MOVEMENT_SPEED * speed;
		if (keys[GLFW_KEY_D])
			offset.x += MOVEMENT_SPEED * speed;
		if (keys[GLFW_KEY_W])
			offset.z += MOVEMENT_SPEED * speed;
		if (keys[GLFW_KEY_S])
			offset.z -= MOVEMENT_SPEED * speed;
		if (keys[GLFW_KEY_LEFT_CONTROL])
			offset.y -= MOVEMENT_SPEED * speed;
		if (keys[GLFW_KEY_SPACE])
			offset.y += MOVEMENT_SPEED * speed;
		if (keys[GLFW_KEY_UP])
			viewOffset.y += speed;
		if (keys[GLFW_KEY_DOWN])
			viewOffset.y -= speed;
		if (keys[GLFW_KEY_LEFT])
			viewOffset.x -= speed;
		if (keys[GLFW_KEY_RIGHT])
			viewOffset.x += speed;

		const bool moved = glm::any(glm::notEqual(offset, vec3(0.0f)));
		const bool rotated = glm::any(glm::notEqual(viewOffset, vec2(0.0f)));

		if (moved)
			move(offset);
		if (rotated)
			rotate(viewOffset);

		return moved | rotated;
	}
} // namespace core