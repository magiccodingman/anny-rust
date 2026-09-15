using System;
using System.IO;
using Anny;
using UnityEngine;

namespace AnnyPlayer
{
    /// <summary>
    /// Runs inside a built player: generate a character from the native library, prove the geometry,
    /// skinning and skeleton came out, print one machine-readable line and quit with a status code.
    /// This is what makes a player build evidence rather than a compilation check.
    /// </summary>
    public sealed class PlayerSmoke : MonoBehaviour
    {
        /// <summary>
        /// Which script backend this player was built with. Resolved by the preprocessor because no
        /// runtime API reports it: IL2CPP defines ENABLE_IL2CPP, Mono does not.
        /// </summary>
        public static string BackendName
        {
            get
            {
#if ENABLE_IL2CPP
                return "il2cpp";
#else
                return "mono";
#endif
            }
        }

        private void Start()
        {
            int exitCode = 1;
            try
            {
                string path = Environment.GetEnvironmentVariable("ANNY_MODEL");
                if (string.IsNullOrEmpty(path) || !File.Exists(path))
                {
                    Debug.LogError("ANNY-PLAYER-SMOKE fail: no model at ANNY_MODEL=" + path);
                }
                else
                {
                    byte[] payload = File.ReadAllBytes(path);
                    AnnyModelAsset asset = AnnyModelAsset.CreateInMemory(payload);
                    GameObject host = new GameObject("anny-player-character");
                    AnnyCharacter character = host.AddComponent<AnnyCharacter>();
                    character.mode = AnnyUpdateMode.Skinned;
                    character.modelAsset = asset;
                    character.Generate();

                    AnnyMeshReport report = character.Report;
                    bool ok = character.GeneratedMesh != null
                        && character.GeneratedMesh.vertexCount > 0
                        && report.SkinWeightsExact
                        && report.SignedVolume > 0f
                        && character.Rig != null
                        && character.Rig.Count == asset.Description.bones
                        && character.Skinned != null;

                    Debug.Log(string.Format(
                        "ANNY-PLAYER-SMOKE {0} vertices={1} meshverts={2} tris={3} bones={4} influences={5} volume={6} runtime={7}",
                        ok ? "ok" : "fail",
                        report.SourceVertices,
                        character.GeneratedMesh != null ? character.GeneratedMesh.vertexCount : -1,
                        character.GeneratedMesh != null ? character.GeneratedMesh.triangles.Length / 3 : -1,
                        character.Rig != null ? character.Rig.Count : -1,
                        report.MaxInfluencesPerVertex,
                        report.SignedVolume,
                        BackendName));

                    exitCode = ok ? 0 : 1;
                }
            }
            catch (Exception error)
            {
                Debug.LogError("ANNY-PLAYER-SMOKE fail: " + error);
            }

            Application.Quit(exitCode);
        }
    }
}
