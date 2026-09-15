using System;
using System.Collections;
using System.IO;
using Anny;
using UnityEngine;
using UnityEngine.Networking;

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
#if UNITY_WEBGL && !UNITY_EDITOR
            StartCoroutine(RunInBrowser());
#else
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
                    exitCode = Report(File.ReadAllBytes(path), path) ? 0 : 1;
                }
            }
            catch (Exception error)
            {
                Debug.LogError("ANNY-PLAYER-SMOKE fail: " + error);
            }

            Debug.Log("ANNY-PLAYER-SMOKE-EXIT " + exitCode);
            Application.Quit(exitCode);
#endif
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        /// <summary>
        /// A browser has no environment variables and no exit code, so the payload comes from the query
        /// string and the machine-readable log line is the result.
        /// </summary>
        private IEnumerator RunInBrowser()
        {
            string url = ModelUrl();
            Debug.Log("ANNY-PLAYER-SMOKE-FETCH " + url);

            using (UnityWebRequest request = UnityWebRequest.Get(url))
            {
                request.timeout = 600;
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogError("ANNY-PLAYER-SMOKE fail: fetch " + request.error);
                }
                else
                {
                    Report(request.downloadHandler.data, url);
                }
            }

            Debug.Log("ANNY-PLAYER-SMOKE-EXIT-DONE");
        }

        private static string ModelUrl()
        {
            string absolute = Application.absoluteURL;
            int query = absolute.IndexOf('?');
            if (query >= 0)
            {
                foreach (string pair in absolute.Substring(query + 1).Split('&'))
                {
                    int equals = pair.IndexOf('=');
                    if (equals > 0 && pair.Substring(0, equals) == "model")
                    {
                        return Uri.UnescapeDataString(pair.Substring(equals + 1));
                    }
                }
            }

            return "model.safetensors";
        }
#endif

        /// <summary>Generates a character from the payload and reports one machine-readable line.</summary>
        private static bool Report(byte[] payload, string origin)
        {
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
                RuntimeName));

            return ok;
        }

        /// <summary>Which scripting backend this player was built with, as reported in the smoke line.</summary>
        private static string RuntimeName
        {
            get
            {
#if UNITY_WEBGL && !UNITY_EDITOR
                return "webgl";
#else
                return BackendName;
#endif
            }
        }
    }
}
