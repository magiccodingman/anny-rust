using Anny;
using Anny.EditorTools;
using NUnit.Framework;
using UnityEditor;
using UnityEngine;

namespace Anny.Tests
{
    /// <summary>
    /// The editor UI is part of the deliverable, so it is checked too: that Unity actually picks the
    /// Anny inspector for a character, and that the bake window hands the baker the options it shows.
    /// </summary>
    public sealed class AnnyEditorUiTests
    {
        private GameObject host;

        [TearDown]
        public void TearDown()
        {
            if (host != null)
            {
                UnityEngine.Object.DestroyImmediate(host);
            }
        }

        [Test]
        public void UnityPicksTheAnnyInspectorForACharacter()
        {
            host = new GameObject("inspector-check");
            AnnyCharacter character = host.AddComponent<AnnyCharacter>();

            Editor editor = Editor.CreateEditor(character);
            try
            {
                Assert.IsInstanceOf<AnnyCharacterEditor>(editor, "the custom inspector should be registered for AnnyCharacter");
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(editor);
            }
        }

        [Test]
        public void TheBakeWindowPassesItsOptionsThroughToTheBaker()
        {
            AnnyBakeWindow window = ScriptableObject.CreateInstance<AnnyBakeWindow>();
            try
            {
                window.expandUvCorners = false;
                window.buildBlendshapes = true;
                window.maxBoneInfluences = 6;

                AnnyMeshOptions options = window.CurrentOptions();
                Assert.IsFalse(options.ExpandUvCorners, "the UV flag should pass through");
                Assert.IsTrue(options.BuildBlendshapes, "the blendshape flag should pass through");
                Assert.AreEqual(6, options.MaxBoneInfluences, "the influence cap should pass through");

                window.payloadPath = "";
                Assert.IsFalse(window.PayloadExists(), "an empty payload should not look bakeable");

                window.payloadPath = "/nonexistent/prepared.safetensors";
                Assert.IsFalse(window.PayloadExists(), "a path that does not exist should not look bakeable");

                if (AnnyTestModels.Available)
                {
                    window.payloadPath = AnnyTestModels.PreparedModelPath;
                    Assert.IsTrue(window.PayloadExists(), "the prepared model should look bakeable");
                }
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(window);
            }
        }
    }
}
