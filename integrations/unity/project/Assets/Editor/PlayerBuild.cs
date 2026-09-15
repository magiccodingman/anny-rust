using System;
using System.IO;
using Anny;
using AnnyPlayer;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

/// <summary>
/// Builds real players and a scene for them to run, so the plugin is exercised under both script
/// backends. The scene is generated rather than hand-authored: it must exist in version control as
/// little more than a character host, and generating it keeps it honest about what it contains.
/// </summary>
public static class PlayerBuild
{
    private const string ScenePath = "Assets/AnnyPlayer/PlayerSmoke.unity";

    [MenuItem("Anny/Build/Linux Mono player")]
    public static void LinuxMono() { Build("mono", ScriptingImplementation.Mono2x); }

    [MenuItem("Anny/Build/Linux IL2CPP player")]
    public static void LinuxIl2cpp() { Build("il2cpp", ScriptingImplementation.IL2CPP); }

    public static void LinuxMonoBatch() { LinuxMono(); }

    public static void LinuxIl2cppBatch() { LinuxIl2cpp(); }

    private static void Build(string label, ScriptingImplementation backend)
    {
        CreateScene();
        PlayerSettings.SetScriptingBackend(NamedBuildTarget.Standalone, backend);

        string root = Directory.GetParent(Application.dataPath).Parent.FullName;   // integrations/unity
        string outDir = Path.Combine(root, "player-" + label);
        Directory.CreateDirectory(outDir);

        BuildPlayerOptions options = new BuildPlayerOptions
        {
            scenes = new[] { ScenePath },
            locationPathName = Path.Combine(outDir, "anny-player"),
            target = BuildTarget.StandaloneLinux64,
            targetGroup = BuildTargetGroup.Standalone,
            options = BuildOptions.None,
        };

        BuildReport report = BuildPipeline.BuildPlayer(options);
        BuildSummary summary = report.summary;
        Debug.Log(string.Format(
            "ANNY-PLAYER-BUILD {0} result={1} errors={2} bytes={3} path={4}",
            label, summary.result, summary.totalErrors, summary.totalSize, options.locationPathName));

        if (summary.result != BuildResult.Succeeded)
        {
            EditorApplication.Exit(1);
        }
    }

    private static void CreateScene()
    {
        UnityEngine.SceneManagement.Scene scene =
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

        GameObject host = new GameObject("anny-player-character");
        AnnyCharacter character = host.AddComponent<AnnyCharacter>();
        character.generateOnAwake = false;   // the smoke behaviour configures it at runtime
        host.AddComponent<PlayerSmoke>();

        EditorSceneManager.SaveScene(scene, ScenePath);
        AssetDatabase.Refresh();
        Debug.Log("ANNY-PLAYER-BUILD scene written to " + ScenePath);
    }
}
