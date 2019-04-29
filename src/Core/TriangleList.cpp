#include "TriangleList.h"
#include "Triangle.cuh"

#include <iostream>

TriangleList::TriangleList()
{
	m_Pool = new ctpl::ThreadPool(ctpl::nr_of_cores);
	// Have at least 1 material
	addMaterial(Material::lambertian(vec3(1.0f), 1.0f));
}

TriangleList::~TriangleList()
{
	for (int i = 0; i < m_Textures.size(); i++)
	{
		delete m_Textures[i];
	}

	delete m_Pool;
}

void TriangleList::addTriangle(vec3 p0, vec3 p1, vec3 p2, vec3 n0, vec3 n1, vec3 n2, unsigned int matIdx, vec2 t0, vec2 t1, vec2 t2)
{
	const auto idx0 = m_Vertices.size();
	const auto idx1 = idx0 + 1;
	const auto idx2 = idx0 + 2;

	m_Vertices.push_back(p0);
	m_Vertices.push_back(p1);
	m_Vertices.push_back(p2);

	m_Normals.push_back(normalize(n0));
	m_Normals.push_back(normalize(n1));
	m_Normals.push_back(normalize(n2));

	m_CenterNormals.push_back((n0 + n1 + n2) / 3.0f);

	m_TexCoords.push_back(t0);
	m_TexCoords.push_back(t1);
	m_TexCoords.push_back(t2);

	m_Indices.emplace_back(idx0, idx1, idx2);

	m_MaterialIdxs.push_back(matIdx);

	m_AABBs.push_back(triangle::getBounds(p0, p1, p2));
}

unsigned int TriangleList::addMaterial(Material mat)
{
	constexpr bool sampleVisibility = true;

	const float alpha = max(0.0f, microfacet::RoughnessToAlpha(mat.roughness));
	m_Microfacets.emplace_back(alpha, alpha, sampleVisibility);

	const unsigned int m = static_cast<unsigned int>(m_Materials.size());

	m_MaterialMutex.lock();
	m_Materials.push_back(mat);
	m_MaterialMutex.unlock();
	return m;
}

void TriangleList::loadModel(const std::string & path, float scale, mat4 mat, int material, bool normalize)
{
	Assimp::Importer importer;
	importer.SetPropertyBool(AI_CONFIG_PP_PTV_NORMALIZE, normalize);
	const aiScene* scene = importer.ReadFile(path,
		aiProcess_FixInfacingNormals |
		aiProcess_GenSmoothNormals |
		aiProcess_JoinIdenticalVertices |
		aiProcess_ImproveCacheLocality |
		aiProcess_LimitBoneWeights |
		aiProcess_FlipWindingOrder |
		aiProcess_RemoveRedundantMaterials |
		aiProcess_Triangulate |
		aiProcess_FindInvalidData |
		aiProcess_PreTransformVertices
	);

	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
	{
		std::cout << "Error Assimp: " << importer.GetErrorString() << std::endl;
		return;
	}

	std::vector<TriangleList::Mesh> meshes = {};

	std::string directory = path.substr(0, path.find_last_of('/'));
	std::mutex meshMutex{};

	processNode(scene->mRootNode, scene, meshes, directory, meshMutex);

	unsigned int offset = m_Vertices.size();

	for (const auto& mesh : meshes)
	{
		for (size_t i = 0; i < mesh.positions.size(); i++)
		{
			const vec4 pos = mat * vec4(mesh.positions[i] * scale, 1.0f);
			const vec4 norm = mat * vec4(mesh.normals[i], 0.0f);

			m_Vertices.push_back(pos);
			m_Normals.push_back(norm);
			m_TexCoords.push_back(mesh.texCoords[i]);
		}
	}

	for (const auto& mesh : meshes)
	{
		for (size_t i = 0; i < mesh.indices.size(); i += 3)
		{
			const auto idx0 = mesh.indices[i + 0];
			const auto idx1 = mesh.indices[i + 1];
			const auto idx2 = mesh.indices[i + 2];

			const auto realIdx0 = idx0 + offset;
			const auto realIdx1 = idx1 + offset;
			const auto realIdx2 = idx2 + offset;

			m_Indices.emplace_back(realIdx0, realIdx1, realIdx2);
			unsigned int matIdx;

			if (material < 0)
				matIdx = mesh.materialIdx;
			else
				matIdx = material;

			if (m_Materials[matIdx].type == Light)
				m_LightIndices.push_back(m_Indices.size() - 1);

			m_MaterialIdxs.push_back(matIdx);
			m_CenterNormals.push_back((m_Normals[realIdx0] + m_Normals[realIdx1] + m_Normals[realIdx2]) / 3.0f);
			m_AABBs.push_back(triangle::getBounds(m_Vertices[realIdx0], m_Vertices[realIdx1], m_Vertices[realIdx2]));
		}

		offset += mesh.positions.size();
	}
}

TriangleList::GPUTextures TriangleList::createTextureBuffer()
{
	std::vector<unsigned int> tDims;
	std::vector<unsigned int> tOffsets;
	std::vector<vec4> tColors;

	for (int i = 0; i < m_Textures.size(); i++)
	{
		unsigned int offset = tColors.size();
		const vec4* buffer = m_Textures[i]->GetTextureBuffer();
		const uint width = m_Textures[i]->GetWidth();
		const uint height = m_Textures[i]->GetHeight();
		for (uint y = 0; y < height; y++)
		{
			for (uint x = 0; x < width; x++)
			{
				tColors.push_back(buffer[x + y * width]);
			}
		}

		tOffsets.push_back(offset);
		tDims.push_back(width);
		tDims.push_back(height);
	}

	GPUTextures buffer;
	buffer.textureDims = std::move(tDims);
	buffer.textureOffsets = std::move(tOffsets);
	buffer.textureColors = std::move(tColors);
	return buffer;
}

void TriangleList::processNode(aiNode * node, const aiScene * scene, std::vector<Mesh> & meshes, const std::string & dir, std::mutex & mMutex)
{
	std::vector<std::future<void>> meshResults;

	for (unsigned int i = 0; i < node->mNumMeshes; i++)
	{
		meshResults.push_back(m_Pool->push([this, i, &mMutex, &meshes, scene, node, &dir](int) {
			aiMesh* mesh = scene->mMeshes[node->mMeshes[i]];
			auto m = processMesh(mesh, scene, dir);
			mMutex.lock();
			meshes.push_back(m);
			mMutex.unlock();
			})
		);
	}

	for (unsigned int i = 0; i < node->mNumChildren; i++)
		processNode(node->mChildren[i], scene, meshes, dir, mMutex);

	for (auto& r : meshResults)
		r.get();
}

TriangleList::Mesh TriangleList::processMesh(aiMesh * mesh, const aiScene * scene, const std::string & dir)
{
	Mesh m{};

	for (unsigned int i = 0; i < mesh->mNumVertices; i++)
	{
		m.positions.push_back(vec3(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z));
		m.normals.push_back(glm::normalize(vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z)));
		if (mesh->mTextureCoords[0]) {
			float one = 1.0f;

			float x = mesh->mTextureCoords[0][i].x;
			float y = mesh->mTextureCoords[0][i].y;

			m.texCoords.emplace_back(x, y);
		}
		else
			m.texCoords.emplace_back(0.5f, 0.5f);
	}

	for (unsigned int i = 0; i < mesh->mNumFaces; i++)
	{
		aiFace& face = mesh->mFaces[i];
		for (unsigned int j = 0; j < face.mNumIndices; j++)
		{
			m.indices.emplace_back(face.mIndices[j]);
		}
	}

	aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

	// return a mesh object created from the extracted mesh data

	auto isNotZero = [](aiColor3D col) {
		return col.r > 0.0001f&& col.g > 0.0001f&& col.b > 0.0001f;
	};

	Material mat;
	aiColor3D color;
	int dTexIdx = -1, nTexIdx = -1, mTexIdx = -1, dispTexIdx = -1;

	for (auto i = 0; i < material->GetTextureCount(aiTextureType_DIFFUSE); i++)
	{
		aiString str;
		material->GetTexture(aiTextureType_DIFFUSE, i, &str);
		std::string path = dir + '/' + str.C_Str();
		dTexIdx = loadTexture(path);
	}

	for (auto i = 0; i < material->GetTextureCount(aiTextureType_NORMALS); i++)
	{
		aiString str;
		material->GetTexture(aiTextureType_NORMALS, i, &str);
		std::string path = dir + '/' + str.C_Str();
		nTexIdx = loadTexture(path);
	}

	for (auto i = 0; i < material->GetTextureCount(aiTextureType_OPACITY); i++)
	{
		aiString str;
		material->GetTexture(aiTextureType_OPACITY, i, &str);
		std::string path = dir + '/' + str.C_Str();
		mTexIdx = loadTexture(path);
	}

	for (auto i = 0; i < material->GetTextureCount(aiTextureType_DISPLACEMENT); i++)
	{
		aiString str;
		material->GetTexture(aiTextureType_DISPLACEMENT, i, &str);
		std::string path = dir + '/' + str.C_Str();
		dispTexIdx = loadTexture(path);
	}

	if (dTexIdx > -1)
	{
		float roughness;
		if (material->Get(AI_MATKEY_SHININESS, roughness) != AI_SUCCESS || roughness < 2.0f)
			mat = Material::lambertian(vec3(1.0f), dTexIdx, nTexIdx, mTexIdx, dispTexIdx);
		else
		{
			mat = Material::lambertian(vec3(1.0f), dTexIdx, nTexIdx, mTexIdx, dispTexIdx);
			mat.roughness = roughness;
			mat.type = GGX;
		}

		m.materialIdx = addMaterial(mat);
		return m;
	}

	if (material->Get(AI_MATKEY_COLOR_EMISSIVE, color) == AI_SUCCESS && isNotZero(color))
	{
		mat = Material::light({ color.r, color.g, color.b });
	}
	else if (material->Get(AI_MATKEY_COLOR_DIFFUSE, color) == AI_SUCCESS && isNotZero(color))
	{
		float roughness = 0.0f;
		aiColor3D specColor;
		if (material->Get(AI_MATKEY_COLOR_SPECULAR, specColor) == AI_SUCCESS)
		{
			const vec3 spec = { specColor.r, specColor.g, specColor.b };
			const vec3 diff = { color.r, color.g, color.b };
			if (dot(spec, spec) > dot(diff, diff))
			{
				color.r = spec.r, color.g = spec.g, color.b = spec.b;
			}
		}

		if (material->Get(AI_MATKEY_SHININESS, roughness) != AI_SUCCESS || roughness < 1)
			mat = Material::lambertian({ color.r, color.g, color.b }, dTexIdx, nTexIdx, mTexIdx, dispTexIdx);
		else
		{
			mat = Material::lambertian({ color.r, color.g, color.b }, dTexIdx, nTexIdx, mTexIdx, dispTexIdx);
			mat.roughness = roughness;
			mat.type = GGX;
		}
	}
	else if (material->Get(AI_MATKEY_COLOR_SPECULAR, color) == AI_SUCCESS && isNotZero(color))
	{
		mat = Material::specular({ color.r, color.g, color.b }, 1.0f);
	}
	else if (material->Get(AI_MATKEY_COLOR_TRANSPARENT, color) == AI_SUCCESS && isNotZero(color))
	{
		float refractIdx;
		if (material->Get(AI_MATKEY_REFRACTI, refractIdx) != AI_SUCCESS)
			refractIdx = 1.2f;
		aiColor3D absorption;
		if (material->Get(AI_MATKEY_COLOR_TRANSPARENT, absorption) != AI_SUCCESS)
			absorption = { 0.0f, 0.0f, 0.0f };

		float roughness = 0.0f;
		if (material->Get(AI_MATKEY_SHININESS, roughness) != AI_SUCCESS || roughness < 1)
		{
			mat = Material::fresnel({ color.r, color.g, color.b }, refractIdx, { absorption.r, absorption.g, absorption.b }, dTexIdx, nTexIdx);
		}
		else
		{
			mat = Material::fresnel({ color.r, color.g, color.b }, refractIdx, { absorption.r, absorption.g, absorption.b }, dTexIdx, nTexIdx);
			mat.roughness = roughness;
			mat.type = FresnelGGX;
		}
	}

	m.materialIdx = addMaterial(mat);

	return m;
}

int TriangleList::overwriteTexture(int idx, const std::string & path)
{
	if (m_Textures.size() <= idx)
		return -1;

	if (m_LoadedTextures.find(path) != m_LoadedTextures.end()) // Texture already in memory
		return m_LoadedTextures.at(path);

	std::string oldTexture;
	for (auto& entry : m_LoadedTextures)
	{
		if (entry.second == idx)
		{
			oldTexture = entry.first;
			break;
		}
	}
	m_LoadedTextures.erase(oldTexture);

	auto* tex = new core::Surface(path.c_str());
	auto* oldTex = m_Textures[idx];
	m_Textures[idx] = tex;
	delete oldTex;
	m_LoadedTextures[path] = idx;
	return idx;
}

int TriangleList::loadTexture(const std::string & path)
{
	if (m_LoadedTextures.find(path) != m_LoadedTextures.end()) // Texture already in memory
		return m_LoadedTextures.at(path);

	try {
		auto* tex = new core::Surface(path.c_str());
		unsigned int idx = m_Textures.size();
		m_LoadedTextures[path] = idx;
		m_Textures.push_back(tex);
		return idx;
	}
	catch (const std::runtime_error & e)
	{
		return -1;
	}
}