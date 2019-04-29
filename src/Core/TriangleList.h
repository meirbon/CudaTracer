#pragma once

#include <string>
#include <vector>
#include <map>
#include <tuple>
#include <future>

#include <glm/glm.hpp>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include "BVH/AABB.h"
#include "Utils/Surface.h"
#include "Material.cuh"
#include "Microfacet.cuh"
#include "Utils/ctpl.h"

using namespace glm;

class TriangleList
{
public:
	struct Mesh
	{
		std::vector<vec3> positions;
		std::vector<vec3> normals;
		std::vector<vec2> texCoords;
		std::vector<unsigned int> indices;
		unsigned int materialIdx;
	};

	struct GPUTextures
	{
		std::vector<unsigned int> textureDims;
		std::vector<unsigned int> textureOffsets;
		std::vector<vec4> textureColors;
	};
public:
	TriangleList();
	~TriangleList();

	void addTriangle(vec3 p0, vec3 p1, vec3 p2, vec3 n0, vec3 n1, vec3 n2, unsigned int matIdx, vec2 t0 = vec2(0.0f), vec2 t1 = vec2(0.0f), vec2 t2 = vec2(0.0f));

	unsigned int addMaterial(Material mat);

	void loadModel(const std::string& path, float scale = 1.0f, mat4 mat = mat4(1.0f), int material = -1, bool normalize = true);

	inline size_t getPrimCount() const
	{
		return m_Indices.size();
	}

	GPUTextures createTextureBuffer();

public:
	std::vector<vec3> m_Vertices{};
	std::vector<vec3> m_Normals{};
	std::vector<vec3> m_CenterNormals{};
	std::vector<vec2> m_TexCoords{};

	std::vector<unsigned int> m_MaterialIdxs{};
	std::vector<unsigned int> m_LightIndices{};

	std::vector<core::Surface*> m_Textures{};
	std::map<std::string, unsigned int> m_LoadedTextures;
	std::vector<Material> m_Materials{};
	std::vector<microfacet::Microfacet> m_Microfacets;

	std::vector<AABB> m_AABBs{};

	std::vector<uvec3> m_Indices{};

	int overwriteTexture(int idx, const std::string& path);
	int loadTexture(const std::string &path);
private:
	std::mutex m_MaterialMutex;
	ctpl::ThreadPool* m_Pool;

	void processNode(aiNode* node, const aiScene* scene, std::vector<Mesh> &meshes, const std::string& dir, std::mutex &mMutex);

	Mesh processMesh(aiMesh* mesh, const aiScene* scene, const std::string& dir);
};
