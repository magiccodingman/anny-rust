using UnityEngine;
using UnityEngine.Rendering;

namespace Anny
{
    /// <summary>
    /// Materials for generated characters. The model payload carries geometry, texture coordinates
    /// and a skeleton but no shading, so a generated character starts on the active render
    /// pipeline's opaque lit shader with the model's UVs; interchange shading authored elsewhere
    /// arrives through the glTF path.
    /// </summary>
    public static class AnnyMaterials
    {
        private static readonly string[] Candidates =
        {
            "Universal Render Pipeline/Lit",
            "HDRP/Lit",
            "Standard",
        };

        /// <summary>Creates an opaque material on the active pipeline's lit shader.</summary>
        public static Material CreateDefault(string name = "Anny Character")
        {
            Shader shader = FindLitShader();
            if (shader == null)
            {
                throw new System.InvalidOperationException(
                    "no lit shader found; pass an explicit material to the character");
            }

            Material material = new Material(shader);
            material.name = name;
            SetColorIfPresent(material, "_BaseColor", Color.white);
            SetColorIfPresent(material, "_Color", Color.white);
            SetFloatIfPresent(material, "_Smoothness", 0.2f);
            return material;
        }

        /// <summary>The lit shader the active pipeline would use.</summary>
        public static Shader FindLitShader()
        {
            for (int i = 0; i < Candidates.Length; i++)
            {
                Shader shader = Shader.Find(Candidates[i]);
                if (shader != null)
                {
                    return shader;
                }
            }

            return Shader.Find("Sprites/Default");
        }

        private static void SetColorIfPresent(Material material, string property, Color value)
        {
            if (material.HasProperty(property))
            {
                material.SetColor(property, value);
            }
        }

        private static void SetFloatIfPresent(Material material, string property, float value)
        {
            if (material.HasProperty(property))
            {
                material.SetFloat(property, value);
            }
        }
    }
}