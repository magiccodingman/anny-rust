using UnityEngine;

namespace Anny
{
    /// <summary>
    /// Drives a <see cref="MeshCollider"/> from the character's generated geometry.
    ///
    /// What the collider follows depends on the update mode, and that is worth being explicit about:
    /// in <see cref="AnnyUpdateMode.Skinned"/> the mesh carries the bind-pose surface, so the collider
    /// is the character's rest shape and does not follow the bones. In <see cref="AnnyUpdateMode.Exact"/>
    /// the mesh carries the evaluated surface, so the collider is the posed character and is refreshed
    /// whenever the pose changes.
    /// </summary>
    [RequireComponent(typeof(AnnyCharacter))]
    [DisallowMultipleComponent]
    public sealed class AnnyMeshCollider : MonoBehaviour
    {
        [Tooltip("Cook a convex hull instead of a triangle mesh. Convex colliders are cheaper and can be used as rigid bodies.")]
        public bool convex;

        [Tooltip("Refresh the collider when the character's geometry changes in Exact mode.")]
        public bool syncOnPoseChange = true;

        private AnnyCharacter character;
        private MeshCollider attached;
        private Mesh lastSynced;
        private int lastSyncedEvaluations = -1;

        /// <summary>The collider this component drives.</summary>
        public MeshCollider Collider
        {
            get { return attached; }
        }

        /// <summary>The mesh the collider currently uses.</summary>
        public Mesh SyncedMesh
        {
            get { return lastSynced; }
        }

        private void OnEnable()
        {
            character = GetComponent<AnnyCharacter>();
            Rebuild();
        }

        /// <summary>
        /// Points the collider at the character's generated mesh, creating the collider on first use.
        /// Re-assigning is what makes Unity re-cook the collision data, so this is called again rather
        /// than assuming an in-place mesh edit is visible to the physics engine.
        /// </summary>
        public bool Rebuild()
        {
            return Rebuild(character != null ? character.GeneratedMesh : null);
        }

        /// <summary>Points the collider at a mesh. Pass null to detach.</summary>
        public bool Rebuild(Mesh source)
        {
            if (source == null)
            {
                return false;
            }

            if (attached == null)
            {
                attached = GetComponent<MeshCollider>();
                if (attached == null)
                {
                    attached = gameObject.AddComponent<MeshCollider>();
                }
            }

            attached.convex = convex;

            if (attached.sharedMesh != source)
            {
                attached.sharedMesh = null;      // force a re-cook even when the reference is unchanged
                attached.sharedMesh = source;
            }
            else
            {
                // Exact mode edits the same Mesh in place, so the physics engine has to be told the
                // collision data is stale; re-assigning is what makes Unity re-cook it.
                attached.sharedMesh = null;
                attached.sharedMesh = source;
            }

            lastSynced = source;
            lastSyncedEvaluations = character != null ? character.Evaluations : -1;
            return true;
        }

        private void LateUpdate()
        {
            if (!syncOnPoseChange || character == null)
            {
                return;
            }

            if (character.mode != AnnyUpdateMode.Exact)
            {
                return;
            }

            // AnnyCharacter updates the generated mesh in place rather than replacing it, so the
            // evaluation counter is the signal that the vertices moved. Comparing mesh references
            // would never fire.
            if (character.GeneratedMesh != lastSynced || character.Evaluations != lastSyncedEvaluations)
            {
                Rebuild();
            }
        }
    }
}
