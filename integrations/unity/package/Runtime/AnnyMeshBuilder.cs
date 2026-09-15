using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace Anny
{
    /// <summary>Options controlling how a native evaluation becomes a Unity <see cref="Mesh"/>.</summary>
    [System.Serializable]
    public sealed class AnnyMeshOptions
    {
        /// <summary>
        /// Splits vertices so each one carries a single UV. Anny stores UVs per face corner, so a
        /// mesh with distinct per-corner UVs cannot be represented without this expansion.
        /// Expanding changes the vertex count and index buffer; disabling it keeps native vertex
        /// order and drops UVs.
        /// </summary>
        public bool ExpandUvCorners { get; set; } = true;

        /// <summary>Upper bound on influences kept per vertex; clamped to 32, Unity's documented ceiling.</summary>
        /// <summary>
        /// Upper bound on influences written per vertex; zero keeps every influence the model
        /// assigns (clamped to <see cref="AnnyMeshBuilder.UnityMaxBoneInfluences"/>). This defaults
        /// to keeping all of them because a dropped influence changes the skinned result.
        /// </summary>
        public int MaxBoneInfluences { get; set; }

        /// <summary>
        /// Build the mesh from the bind-pose geometry (rest_vertices) rather than the evaluated
        /// vertices. A skinned renderer needs this: the mesh is the input to skinning and the bones
        /// are the transform, so uploading an already-posed mesh skins the character a second time.
        /// Anny's default evaluation is not a bind pose, so the two differ by centimetres.
        /// </summary>
        public bool UseBindPoseGeometry { get; set; }

        /// <summary>Builds morph targets from the model's blendshape stack.</summary>
        public bool BuildBlendshapes { get; set; }

        /// <summary>Optional subset of blendshape labels to build; null considers every target.</summary>
        public string[] BlendshapeLabels { get; set; }

        /// <summary>Skips blendshape frames whose deltas are all zero.</summary>
        public bool SkipEmptyBlendshapes { get; set; } = true;

        /// <summary>Recalculates normals from the built triangulation.</summary>
        public bool RecalculateNormals { get; set; } = true;

        /// <summary>Recalculates bounds from the built positions.</summary>
        public bool RecalculateBounds { get; set; } = true;

        /// <summary>Builds tangents from the UV set; requires UVs and normals.</summary>
        public bool RecalculateTangents { get; set; }
    }

    /// <summary>Measured facts about one mesh build, including any lossy step taken.</summary>
    public sealed class AnnyMeshReport
    {
        public int SourceVertices;
        public int MeshVertices;
        public int Faces;
        public int QuadFaces;
        public int Triangles;
        public int BoneInfluenceWidth;
        public int MaxBoneInfluences;

        public int VerticesWithTruncatedInfluences;
        public float LargestDroppedWeight;

        /// <summary>
        /// Vertices whose native influences were not already in descending weight order. Unity
        /// requires descending order and reports an error otherwise, so the packer sorts; this
        /// records how often that was actually needed rather than assuming it.
        /// </summary>
        public int VerticesWithUnsortedNativeWeights;
        public int MaxInfluencesPerVertex;
        public bool UvExpanded;
        public float SignedVolume;
        public string[] BlendShapeNames;

        /// <summary>True when every influence the model assigns was preserved exactly.</summary>
        public bool SkinWeightsExact
        {
            get { return VerticesWithTruncatedInfluences == 0; }
        }
    }

    /// <summary>
    /// Wraps the model's morph-target stack for either precision so the mesh builder has one
    /// accessor. Frames are read element-wise, so nothing is densified into the wrong precision.
    /// </summary>
    public sealed class AnnyBlendshapeTensor
    {
        public int[] Shape;
        public string[] Labels;
        public double[] Data64;
        public float[] Data32;

        public bool IsSingle
        {
            get { return Data64 == null; }
        }

        /// <summary>Number of morph targets, or zero when the model has none.</summary>
        public int Count
        {
            get { return Shape == null ? 0 : Shape[0]; }
        }

        /// <summary>Vertices each frame addresses.</summary>
        public int VertexCount
        {
            get { return Shape == null ? 0 : Shape[1]; }
        }

        public double At(int index)
        {
            return IsSingle ? Data32[index] : Data64[index];
        }

        public static AnnyBlendshapeTensor FromModel(AnnyModel model)
        {
            if (model == null || !model.HasTensor("blendshapes"))
            {
                return null;
            }

            AnnyTensor tensor = model.Tensor("blendshapes");
            return new AnnyBlendshapeTensor
            {
                Shape = tensor.Shape,
                Data64 = tensor.Data,
                Labels = model.Description.BlendShapeLabels,
            };
        }

        public static AnnyBlendshapeTensor FromModel(AnnyModelF32 model)
        {
            if (model == null || !model.HasTensor("blendshapes"))
            {
                return null;
            }

            AnnyTensorF32 tensor = model.Tensor("blendshapes");
            return new AnnyBlendshapeTensor
            {
                Shape = tensor.Shape,
                Data32 = tensor.Data,
                Labels = model.Description.BlendShapeLabels,
            };
        }
    }

    /// <summary>A built mesh together with its bind poses and measurement report.</summary>
    public sealed class AnnyMeshResult
    {
        public Mesh Mesh;
        public Matrix4x4[] BindPoses;
        public AnnyMeshReport Report;

        /// <summary>
        /// Maps each mesh vertex to the source vertex it came from, or null when the mesh kept the
        /// native vertex order. When UV corners are expanded one native vertex backs several mesh
        /// vertices, and anything that indexes native arrays per mesh vertex — the exact vertex
        /// update, weights, morph deltas — has to go through this map.
        /// </summary>
        public int[] CornerSource;
    }

    /// <summary>
    /// Builds Unity meshes directly from native tensors: posed positions, triangles
    /// (triangulating quad faces), per-corner UVs, single-precision skinning weights, bind poses
    /// from the rest bone matrices, and optional morph targets.
    /// </summary>
    public static class AnnyMeshBuilder
    {
        /// <summary>Unity's documented ceiling for single-precision skinning weights per vertex.</summary>
        public const int UnityMaxBoneInfluences = 32;

        /// <summary>Comfortable default for the bone-weight array; matches typical hardware skinning.</summary>
        /// <summary>Keeps every influence the model assigns.</summary>
        public const int DefaultBoneInfluences = 0;

        /// <summary>Builds a mesh plus bind poses from an f64 evaluation.</summary>
        public static AnnyMeshResult Build(AnnyModel model, AnnyOutput output, AnnyMeshOptions options = null)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            options = options ?? new AnnyMeshOptions();
            AnnyTensor positions = output.Tensor(AnnyOutput.Vertices);
            AnnyTensor bind = output.Tensor(AnnyOutput.RestBonePoses);
            AnnyTensor faces = model.Tensor("faces");
            AnnyTensor skinWeights = model.Tensor("vertex_bone_weights");
            AnnyTensor skinBones = model.Tensor("vertex_bone_indices");

            int[] faceIndex = faces.ToIndices();
            int faceWidth = faces.Shape[1];
            bool expand = options.ExpandUvCorners
                && model.HasTensor("texture_coordinates")
                && model.HasTensor("face_texture_coordinate_indices");
            AnnyTensor faceUv = expand ? model.Tensor("face_texture_coordinate_indices") : null;
            AnnyTensor uvs = expand ? model.Tensor("texture_coordinates") : null;

            return Assemble(
                positions.Data,
                positions.Shape[1],
                bind.Data,
                bind.Shape[1],
                faceIndex,
                faceWidth,
                skinWeights.Data,
                skinWeights.Shape[1],
                skinBones.ToIndices(),
                expand ? faceUv.ToIndices() : null,
                expand ? uvs.Data : null,
                options,
                options.BuildBlendshapes ? AnnyBlendshapeTensor.FromModel(model) : null,
                faces.Length);
        }

        /// <summary>Builds a mesh plus bind poses from an f32 evaluation.</summary>
        public static AnnyMeshResult Build(AnnyModelF32 model, AnnyOutputF32 output, AnnyMeshOptions options = null)
        {
            if (model == null)
            {
                throw new ArgumentNullException(nameof(model));
            }

            if (output == null)
            {
                throw new ArgumentNullException(nameof(output));
            }

            options = options ?? new AnnyMeshOptions();
            AnnyTensorF32 positions = output.Tensor(
                options.UseBindPoseGeometry ? AnnyOutputF32.RestVertices : AnnyOutputF32.Vertices);
            AnnyTensorF32 bind = output.Tensor(AnnyOutputF32.RestBonePoses);
            AnnyTensorF32 faces = model.Tensor("faces");
            AnnyTensorF32 skinWeights = model.Tensor("vertex_bone_weights");
            AnnyTensorF32 skinBones = model.Tensor("vertex_bone_indices");

            int[] faceIndex = faces.ToIndices();
            int faceWidth = faces.Shape[1];
            bool expand = options.ExpandUvCorners
                && model.HasTensor("texture_coordinates")
                && model.HasTensor("face_texture_coordinate_indices");
            AnnyTensorF32 faceUv = expand ? model.Tensor("face_texture_coordinate_indices") : null;
            AnnyTensorF32 uvs = expand ? model.Tensor("texture_coordinates") : null;

            double[] widePositions = new double[positions.Data.Length];
            for (int i = 0; i < widePositions.Length; i++)
            {
                widePositions[i] = positions.Data[i];
            }

            double[] wideBind = new double[bind.Data.Length];
            for (int i = 0; i < wideBind.Length; i++)
            {
                wideBind[i] = bind.Data[i];
            }

            double[] wideWeights = new double[skinWeights.Data.Length];
            for (int i = 0; i < wideWeights.Length; i++)
            {
                wideWeights[i] = skinWeights.Data[i];
            }

            double[] wideUvs = null;
            if (expand)
            {
                wideUvs = new double[uvs.Data.Length];
                for (int i = 0; i < wideUvs.Length; i++)
                {
                    wideUvs[i] = uvs.Data[i];
                }
            }

            return Assemble(
                widePositions,
                positions.Shape[1],
                wideBind,
                bind.Shape[1],
                faceIndex,
                faceWidth,
                wideWeights,
                skinWeights.Shape[1],
                skinBones.ToIndices(),
                expand ? faceUv.ToIndices() : null,
                wideUvs,
                options,
                options.BuildBlendshapes ? AnnyBlendshapeTensor.FromModel(model) : null,
                faces.Length);
        }

        /// <summary>
        /// Overwrites a mesh's vertices in place from a fresh evaluation, reusing the mesh's
        /// existing index stream. This is the exactness path: what lands in the mesh is the
        /// native posed array narrowed to float, with no interpolation in between, so a caller can
        /// compare the two arrays element by element.
        /// </summary>
        public static void UpdatePosedVertices(
            Mesh mesh, IAnnyEvaluation evaluation, int[] cornerSource, bool recalculateBounds)
        {
            if (mesh == null)
            {
                throw new ArgumentNullException(nameof(mesh));
            }

            AnnyTensorF32 posed = evaluation.Tensor(AnnyOutputF32.Vertices);
            int sourceCount = posed.Shape[1];
            int target = cornerSource != null ? cornerSource.Length : sourceCount;

            if (target != mesh.vertexCount)
            {
                throw new InvalidOperationException(
                    "the evaluation maps to " + target + " vertices but the mesh has " + mesh.vertexCount
                    + "; the mesh was built from a different evaluation shape");
            }

            Vector3[] vertices = new Vector3[target];
            for (int i = 0; i < target; i++)
            {
                int source = cornerSource != null ? cornerSource[i] : i;
                vertices[i] = AnnyRepresentation.ToUnityPosition(
                    posed.Data[source * 3], posed.Data[source * 3 + 1], posed.Data[source * 3 + 2]);
            }

            mesh.SetVertices(vertices);
            if (recalculateBounds)
            {
                mesh.RecalculateBounds();
            }
        }

        /// <summary>Converts a flat native vertex array into Unity positions.</summary>
        public static Vector3[] ToPositions(float[] data, int vertexCount)
        {
            Vector3[] vertices = new Vector3[vertexCount];
            for (int i = 0; i < vertexCount; i++)
            {
                vertices[i] = AnnyRepresentation.ToUnityPosition(data[i * 3], data[i * 3 + 1], data[i * 3 + 2]);
            }

            return vertices;
        }

        private static AnnyMeshResult Assemble(
            double[] positions,
            int vertexCount,
            double[] bindMatrices,
            int boneCount,
            int[] faces,
            int faceWidth,
            double[] weights,
            int influenceWidth,
            int[] boneIndices,
            int[] faceUv,
            double[] uvs,
            AnnyMeshOptions options,
            AnnyBlendshapeTensor blendshapes,
            int cornerCount)
        {
            if (faceWidth != 3 && faceWidth != 4)
            {
                throw new InvalidOperationException("faces must have 3 or 4 corners, found " + faceWidth);
            }

            if (positions.Length < vertexCount * 3)
            {
                throw new InvalidOperationException("the posed vertex array is shorter than the model's vertex count");
            }

            int faceCount = cornerCount / faceWidth;
            AnnyMeshReport report = new AnnyMeshReport
            {
                SourceVertices = vertexCount,
                Faces = faceCount,
                QuadFaces = faceWidth == 4 ? faceCount : 0,
                BoneInfluenceWidth = influenceWidth,
                MaxBoneInfluences = Mathf.Clamp(
                    options.MaxBoneInfluences <= 0 ? UnityMaxBoneInfluences : options.MaxBoneInfluences,
                    1,
                    UnityMaxBoneInfluences),
            };

            // With per-corner UVs every face corner becomes its own mesh vertex, so a corner's
            // mesh index is its corner index and its source vertex is faces[corner]. Without the
            // expansion the mesh keeps the native vertex order and no UV set is emitted.
            bool expand = faceUv != null && uvs != null;
            int[] sourceOf = expand ? faces : null;
            int meshVertexCount = expand ? faces.Length : vertexCount;
            report.UvExpanded = expand;
            report.MeshVertices = meshVertexCount;

            Vector3[] meshPositions = new Vector3[meshVertexCount];
            if (expand)
            {
                for (int corner = 0; corner < meshVertexCount; corner++)
                {
                    int source = faces[corner];
                    meshPositions[corner] = AnnyRepresentation.ToUnityPosition(
                        positions[source * 3], positions[source * 3 + 1], positions[source * 3 + 2]);
                }
            }
            else
            {
                for (int v = 0; v < meshVertexCount; v++)
                {
                    meshPositions[v] = AnnyRepresentation.ToUnityPosition(
                        positions[v * 3], positions[v * 3 + 1], positions[v * 3 + 2]);
                }
            }

            int[] triangles = Triangulate(faces, faceWidth, faceCount, expand);
            report.Triangles = triangles.Length / 3;

            Vector2[] meshUvs = null;
            if (expand)
            {
                meshUvs = new Vector2[meshVertexCount];
                for (int corner = 0; corner < meshVertexCount; corner++)
                {
                    int uvIndex = faceUv[corner];
                    meshUvs[corner] = new Vector2((float)uvs[uvIndex * 2], (float)uvs[uvIndex * 2 + 1]);
                }
            }

            PackedSkin skin = BuildBoneWeights(
                weights, influenceWidth, boneIndices, meshVertexCount, sourceOf, report);

            Matrix4x4[] bindPoses = new Matrix4x4[boneCount];
            for (int b = 0; b < boneCount; b++)
            {
                bindPoses[b] = AnnyRepresentation.ToUnityMatrix(bindMatrices, b * 16).inverse;
            }

            Mesh mesh = new Mesh();
            mesh.name = "Anny Character";
            mesh.indexFormat = meshVertexCount > 65000 ? IndexFormat.UInt32 : IndexFormat.UInt16;
            mesh.SetVertices(meshPositions);
            mesh.SetTriangles(triangles, 0, false);
            if (meshUvs != null)
            {
                mesh.SetUVs(0, meshUvs);
            }

            Unity.Collections.NativeArray<byte> perVertex = new Unity.Collections.NativeArray<byte>(
                skin.BonesPerVertex, Unity.Collections.Allocator.Temp);
            Unity.Collections.NativeArray<BoneWeight1> flat = new Unity.Collections.NativeArray<BoneWeight1>(
                skin.Weights, Unity.Collections.Allocator.Temp);
            try
            {
                mesh.SetBoneWeights(perVertex, flat);
            }
            finally
            {
                perVertex.Dispose();
                flat.Dispose();
            }

            mesh.bindposes = bindPoses;

            if (options.RecalculateNormals)
            {
                mesh.RecalculateNormals();
            }

            if (options.RecalculateTangents)
            {
                mesh.RecalculateTangents();
            }

            if (options.RecalculateBounds)
            {
                mesh.RecalculateBounds();
            }

            report.SignedVolume = SignedVolume(meshPositions, triangles);
            if (blendshapes != null)
            {
                BuildBlendshapes(mesh, blendshapes, options, sourceOf, report);
            }

            return new AnnyMeshResult
            {
                Mesh = mesh,
                BindPoses = bindPoses,
                Report = report,
                CornerSource = sourceOf,
            };
        }

        /// <summary>
        /// Triangulates the face list. Anny faces are ordered corners, so a quad splits into
        /// (0,1,2) and (0,2,3).
        /// <para>
        /// The corner order is preserved. The Anny-to-Unity mapping (x, y, z) -&gt; (x, z, -y) is a
        /// proper rotation (its basis matrix has determinant +1, it is a quarter turn about X), so
        /// it does not mirror the geometry and face winding already points outward. FacesPointOutward
        /// checks that with the signed volume rather than trusting this reasoning.
        /// </para>
        /// </summary>
        public static int[] Triangulate(int[] faces, int faceWidth, int faceCount, bool expanded)
        {
            int[] triangles = new int[faceCount * (faceWidth == 4 ? 6 : 3)];
            int at = 0;
            for (int f = 0; f < faceCount; f++)
            {
                int c = f * faceWidth;
                int v0 = expanded ? c : faces[c];
                int v1 = expanded ? c + 1 : faces[c + 1];
                int v2 = expanded ? c + 2 : faces[c + 2];
                triangles[at++] = v0;
                triangles[at++] = v1;
                triangles[at++] = v2;
                if (faceWidth == 4)
                {
                    int v3 = expanded ? c + 3 : faces[c + 3];
                    triangles[at++] = v0;
                    triangles[at++] = v2;
                    triangles[at++] = v3;
                }
            }

            return triangles;
        }

        /// <summary>The flat bone-weight layout Unity expects, with a count byte per vertex.</summary>
        public struct PackedSkin
        {
            public byte[] BonesPerVertex;
            public BoneWeight1[] Weights;
        }

        /// <summary>
        /// Packs native per-vertex influences into Unity's flat layout, keeping the nonzero
        /// influences in native order. Influences beyond the limit are dropped and the remainder
        /// renormalized so the weights stay a convex combination; the report records every vertex
        /// where that happened.
        /// </summary>
        public static PackedSkin BuildBoneWeights(
            double[] weights,
            int width,
            int[] boneIndices,
            int meshVertexCount,
            int[] sourceOf,
            AnnyMeshReport report)
        {
            int limit = report.MaxBoneInfluences;
            if (limit <= 0)
            {
                limit = width;
            }

            if (limit > 32)
            {
                limit = 32;
            }

            List<BoneWeight1> flat = new List<BoneWeight1>(meshVertexCount * 4);
            byte[] perVertex = new byte[meshVertexCount];
            double[] all = new double[width];
            int[] allBones = new int[width];

            for (int v = 0; v < meshVertexCount; v++)
            {
                int source = sourceOf == null ? v : sourceOf[v];

                // Collect every influence the model actually assigns. Zero-weight entries are not
                // influences: adding them back would change skinning, because a zero term still
                // rounds the sum.
                int count = 0;
                double sum = 0.0;
                double previous = double.PositiveInfinity;
                bool ordered = true;
                for (int i = 0; i < width; i++)
                {
                    double w = weights[source * width + i];
                    if (w == 0.0)
                    {
                        continue;
                    }

                    if (w > previous)
                    {
                        ordered = false;
                    }

                    previous = w;
                    sum += w;
                    all[count] = w;
                    allBones[count] = boneIndices[source * width + i];
                    count++;
                }

                if (!ordered)
                {
                    report.VerticesWithUnsortedNativeWeights++;
                }

                // Descending insertion sort: Unity requires Mesh.boneWeights in descending order,
                // and it is what makes the truncation below drop the smallest influences.
                for (int i = 1; i < count; i++)
                {
                    double w = all[i];
                    int bone = allBones[i];
                    int j = i - 1;
                    while (j >= 0 && all[j] < w)
                    {
                        all[j + 1] = all[j];
                        allBones[j + 1] = allBones[j];
                        j--;
                    }

                    all[j + 1] = w;
                    allBones[j + 1] = bone;
                }

                int keep = count < limit ? count : limit;
                double dropped = 0.0;
                for (int i = keep; i < count; i++)
                {
                    dropped += all[i];
                }

                if (dropped > 0.0)
                {
                    report.VerticesWithTruncatedInfluences++;
                    if ((float)dropped > report.LargestDroppedWeight)
                    {
                        report.LargestDroppedWeight = (float)dropped;
                    }
                }

                if (keep > report.MaxInfluencesPerVertex)
                {
                    report.MaxInfluencesPerVertex = keep;
                }

                if (keep == 0)
                {
                    // The model contract binds helpers with no weights to bone 0 at full weight.
                    perVertex[v] = 1;
                    flat.Add(new BoneWeight1 { boneIndex = 0, weight = 1f });
                    continue;
                }

                double scale = dropped > 0.0 && sum > dropped ? sum / (sum - dropped) : 1.0;
                perVertex[v] = (byte)keep;
                for (int i = 0; i < keep; i++)
                {
                    flat.Add(new BoneWeight1 { boneIndex = allBones[i], weight = (float)(all[i] * scale) });
                }
            }

            return new PackedSkin { BonesPerVertex = perVertex, Weights = flat.ToArray() };
        }

        /// <summary>Computes the signed volume of a triangle soup; positive means outward winding.</summary>
        public static float SignedVolume(Vector3[] positions, int[] triangles)
        {
            double total = 0.0;
            for (int i = 0; i < triangles.Length; i += 3)
            {
                Vector3 a = positions[triangles[i]];
                Vector3 b = positions[triangles[i + 1]];
                Vector3 c = positions[triangles[i + 2]];
                total += Vector3.Dot(a, Vector3.Cross(b, c)) / 6.0;
            }

            return (float)total;
        }

        private static void BuildBlendshapes(
            Mesh mesh, AnnyBlendshapeTensor shapes, AnnyMeshOptions options, int[] sourceOf, AnnyMeshReport report)
        {
            string[] labels = shapes.Labels;
            int count = shapes.Count;
            int vertexCount = shapes.VertexCount;
            if (mesh.vertexCount != vertexCount && sourceOf == null)
            {
                throw new InvalidOperationException(
                    "a mesh without UV expansion must have as many vertices as the model has");
            }

            HashSet<string> subset = options.BlendshapeLabels != null
                ? new HashSet<string>(options.BlendshapeLabels)
                : null;
            List<string> built = new List<string>();
            Vector3[] deltas = new Vector3[mesh.vertexCount];

            for (int s = 0; s < count; s++)
            {
                string label = labels != null && s < labels.Length ? labels[s] : "blendshape_" + s;
                if (subset != null && !subset.Contains(label))
                {
                    continue;
                }

                bool any = false;
                for (int v = 0; v < mesh.vertexCount; v++)
                {
                    int source = sourceOf == null ? v : sourceOf[v];
                    int offset = (s * vertexCount + source) * 3;
                    Vector3 delta = AnnyRepresentation.ToUnityPosition(
                        shapes.At(offset), shapes.At(offset + 1), shapes.At(offset + 2));
                    deltas[v] = delta;
                    any |= delta != Vector3.zero;
                }

                if (!any && options.SkipEmptyBlendshapes)
                {
                    continue;
                }

                mesh.AddBlendShapeFrame(label, 100f, deltas, null, null);
                built.Add(label);
            }

            report.BlendShapeNames = built.ToArray();
        }
    }
}