#pragma once

#include <GL/glew.h>
#include <cuda_runtime.h>

class CudaRenderer
{
public:
	explicit CudaRenderer(int width, int height);
	~CudaRenderer();

	void setDimensions(int width, int height);

	void draw();

	inline cudaArray* getCudaArray()
	{
		return m_CudaFBArray;
	}

private:
	int m_Width, m_Height;
	GLuint m_Renderbuffer, m_Framebuffer;

	cudaGraphicsResource* m_CudaFB = nullptr;
	cudaArray* m_CudaFBArray = nullptr;
};
