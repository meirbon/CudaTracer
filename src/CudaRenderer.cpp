#include "CudaRenderer.h"
#include "CudaAssert.h"
#include <cuda_gl_interop.h>

CudaRenderer::CudaRenderer(int width, int height)
{
	glCreateRenderbuffers(1, &m_Renderbuffer);
	glCreateFramebuffers(1, &m_Framebuffer);

	glNamedFramebufferRenderbuffer(m_Framebuffer, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, m_Renderbuffer);

	setDimensions(width, height);
}

CudaRenderer::~CudaRenderer()
{
	cudaError err;

	if (m_CudaFB != nullptr)
		err = cuda(GraphicsUnregisterResource(m_CudaFB));

	glDeleteRenderbuffers(1, &m_Renderbuffer);
	glDeleteFramebuffers(1, &m_Framebuffer);
}

void CudaRenderer::setDimensions(int width, int height)
{
	cudaDeviceSynchronize();
	cudaError err;

	m_Width = width;
	m_Height = height;

	if (m_CudaFB != nullptr)
		err = cuda(GraphicsUnregisterResource(m_CudaFB));

	glNamedRenderbufferStorage(m_Renderbuffer, GL_RGBA32F, m_Width, m_Height);

	const auto flags = cudaGraphicsRegisterFlagsSurfaceLoadStore | cudaGraphicsRegisterFlagsWriteDiscard;
	err = cuda(GraphicsGLRegisterImage(&m_CudaFB, m_Renderbuffer, GL_RENDERBUFFER, flags));
	err = cuda(GraphicsMapResources(1, &m_CudaFB, 0));
	err = cuda(GraphicsSubResourceGetMappedArray(&m_CudaFBArray, m_CudaFB, 0, 0));
	err = cuda(GraphicsUnmapResources(1, &m_CudaFB, 0));
}

void CudaRenderer::draw()
{
	glBlitNamedFramebuffer(m_Framebuffer, 0, 0, 0, m_Width, m_Height, 0, m_Height, m_Width, 0, GL_COLOR_BUFFER_BIT, GL_NEAREST);
}