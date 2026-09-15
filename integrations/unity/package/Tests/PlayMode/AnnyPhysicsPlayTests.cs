using System.Collections;
using System.IO;
using Anny;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace Anny.Tests
{
    /// <summary>
    /// Physics integration, checked against the mesh the collider was cooked from rather than against
    /// "a collider exists". A MeshCollider holding stale collision data still answers raycasts
    /// perfectly well, so the only assertion worth making compares the hit with the geometry.
    /// </summary>
    public sealed class AnnyPhysicsPlayTests
    {
        private GameObject host;
        private AnnyCharacter character;

        private static string ModelPath
        {
            get
            {
                string fromEnvironment = System.Environment.GetEnvironmentVariable("ANNY_MODEL");
                if (!string.IsNullOrEmpty(fromEnvironment) && File.Exists(fromEnvironment))
                {
                    return fromEnvironment;
                }

                DirectoryInfo project = new DirectoryInfo(Application.dataPath).Parent;
                string candidate = Path.GetFullPath(
                    Path.Combine(project.FullName, "..", "..", "..", "output", "ci-model.safetensors"));
                return File.Exists(candidate) ? candidate : null;
            }
        }

        [SetUp]
        public void SetUp()
        {
            if (ModelPath == null)
            {
                Assert.Ignore("prepared model not found; set ANNY_MODEL");
            }
        }

        [TearDown]
        public void TearDown()
        {
            if (host != null)
            {
                Object.Destroy(host);
                host = null;
                character = null;
            }
        }

        private void Spawn(AnnyUpdateMode mode)
        {
            host = new GameObject("anny-physics");
            host.SetActive(false);
            character = host.AddComponent<AnnyCharacter>();
            character.generateOnAwake = false;
            character.mode = mode;
            character.modelAsset = AnnyModelAsset.CreateInMemory(File.ReadAllBytes(ModelPath));
            host.SetActive(true);
        }

        [UnityTest]
        public IEnumerator TheColliderUsesTheGeneratedMeshAndMatchesItsBounds()
        {
            Spawn(AnnyUpdateMode.Skinned);
            character.Generate();

            AnnyMeshCollider attached = host.AddComponent<AnnyMeshCollider>();
            Physics.SyncTransforms();
            yield return null;
            Physics.SyncTransforms();

            Assert.IsNotNull(attached.Collider, "adding the component should create a collider");
            Assert.AreSame(character.GeneratedMesh, attached.Collider.sharedMesh, "the collider should use the generated mesh");
            Assert.IsFalse(attached.Collider.convex, "the default is a triangle mesh collider");

            Bounds fromMesh = character.GeneratedMesh.bounds;
            Bounds fromCollider = attached.Collider.bounds;
            Assert.Less(
                (fromMesh.size - fromCollider.size).magnitude,
                1e-3f,
                string.Format("collider size {0} vs mesh size {1}", fromCollider.size, fromMesh.size));
            Assert.Less(
                (fromMesh.center - fromCollider.center).magnitude,
                1e-3f,
                string.Format("collider centre {0} vs mesh centre {1}", fromCollider.center, fromMesh.center));
        }

        [UnityTest]
        public IEnumerator ARaycastHitsTheSurfaceTheMeshDefines()
        {
            Spawn(AnnyUpdateMode.Skinned);
            character.Generate();
            host.AddComponent<AnnyMeshCollider>();
            Physics.SyncTransforms();
            yield return null;
            Physics.SyncTransforms();

            Bounds bounds = character.GeneratedMesh.bounds;
            Vector3 origin = bounds.center + Vector3.up * (bounds.extents.y + 1f);
            Vector3 direction = Vector3.down;

            float fromGeometry = NearestHit(character.GeneratedMesh, origin, direction);
            Assert.IsTrue(fromGeometry < float.PositiveInfinity, "the vertical ray should cross the mesh");

            RaycastHit hit;
            bool touched = Physics.Raycast(origin, direction, out hit, bounds.size.y + 2f);
            Debug.Log(string.Format("ANNY-PHYSICS-RAYCAST collider={0:0.########} mesh={1:0.########} delta={2:0.########}",
                hit.distance, fromGeometry, Mathf.Abs(hit.distance - fromGeometry)));
            Assert.IsTrue(touched, "the ray should hit the collider");
            Assert.Less(
                Mathf.Abs(hit.distance - fromGeometry),
                1e-3f,
                string.Format("raycast hit at {0:0.######} m, the mesh geometry says {1:0.######} m", hit.distance, fromGeometry));
        }

        [UnityTest]
        public IEnumerator ExactModeRefreshesTheColliderWhenThePoseChanges()
        {
            Spawn(AnnyUpdateMode.Exact);
            character.Generate();

            AnnyMeshCollider attached = host.AddComponent<AnnyMeshCollider>();
            Physics.SyncTransforms();
            yield return null;
            Physics.SyncTransforms();

            Bounds before = attached.Collider.bounds;
            int evaluationsBefore = character.Evaluations;

            character.SetPhenotype("gender", 1f);
            character.Apply();
            yield return null;
            Physics.SyncTransforms();

            Assert.Greater(character.Evaluations, evaluationsBefore, "the pose change should have re-evaluated");

            Bounds after = attached.Collider.bounds;
            float movement = (after.center - before.center).magnitude + (after.size - before.size).magnitude;
            Assert.Greater(
                movement,
                1e-5f,
                "the collider is stale: Exact mode edits the mesh in place, so it has to be re-cooked");
        }

        /// <summary>Distance at which a ray first crosses the mesh, by ray/triangle intersection.</summary>
        private static float NearestHit(Mesh mesh, Vector3 origin, Vector3 direction)
        {
            Vector3[] vertices = mesh.vertices;
            int[] triangles = mesh.triangles;
            float nearest = float.PositiveInfinity;

            for (int i = 0; i < triangles.Length; i += 3)
            {
                float distance;
                if (Intersects(origin, direction, vertices[triangles[i]], vertices[triangles[i + 1]], vertices[triangles[i + 2]], out distance)
                    && distance < nearest)
                {
                    nearest = distance;
                }
            }

            return nearest;
        }

        private static bool Intersects(Vector3 origin, Vector3 direction, Vector3 a, Vector3 b, Vector3 c, out float distance)
        {
            distance = 0f;
            Vector3 edge1 = b - a;
            Vector3 edge2 = c - a;
            Vector3 pvec = Vector3.Cross(direction, edge2);
            float determinant = Vector3.Dot(edge1, pvec);
            if (Mathf.Abs(determinant) < 1e-12f)
            {
                return false;
            }

            float inverse = 1f / determinant;
            Vector3 tvec = origin - a;
            float u = Vector3.Dot(tvec, pvec) * inverse;
            if (u < 0f || u > 1f)
            {
                return false;
            }

            Vector3 qvec = Vector3.Cross(tvec, edge1);
            float v = Vector3.Dot(direction, qvec) * inverse;
            if (v < 0f || u + v > 1f)
            {
                return false;
            }

            float t = Vector3.Dot(edge2, qvec) * inverse;
            if (t <= 0f)
            {
                return false;
            }

            distance = t;
            return true;
        }
    }
}
