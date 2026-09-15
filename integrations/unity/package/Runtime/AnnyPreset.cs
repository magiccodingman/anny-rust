using System.Collections.Generic;
using UnityEngine;

namespace Anny
{
    /// <summary>
    /// A saved character configuration: update mode, precision, phenotype sliders and the mesh
    /// options that change geometry. Stored as an ordinary Unity asset, so a preset is data the
    /// project owns rather than something a script has to reconstruct.
    /// </summary>
    [CreateAssetMenu(fileName = "AnnyPreset", menuName = "Anny/Preset", order = 1)]
    public sealed class AnnyPreset : ScriptableObject
    {
        [Tooltip("How the character is driven at runtime.")]
        public AnnyUpdateMode mode = AnnyUpdateMode.Skinned;

        [Tooltip("Evaluate in single precision, which is the default and the faster path.")]
        public bool singlePrecision = true;

        [Tooltip("Phenotype slider values, by label.")]
        public List<AnnySliderEntry> phenotype = new List<AnnySliderEntry>();

        [Header("Mesh options that change geometry")]
        [Tooltip("0 keeps every native influence; a positive value truncates to the largest N.")]
        public int maxBoneInfluences;

        public bool buildBlendshapes;

        public bool expandUvCorners = true;

        /// <summary>Copy the configuration out of a character.</summary>
        public void CaptureFrom(AnnyCharacter character)
        {
            if (character == null) throw new System.ArgumentNullException(nameof(character));

            mode = character.mode;
            singlePrecision = character.singlePrecision;
            phenotype = new List<AnnySliderEntry>(character.phenotype);

            AnnyMeshOptions options = character.meshOptions;
            maxBoneInfluences = options.MaxBoneInfluences;
            buildBlendshapes = options.BuildBlendshapes;
            expandUvCorners = options.ExpandUvCorners;
        }

        /// <summary>Write the configuration back onto a character. Regenerate afterwards to apply it.</summary>
        public void ApplyTo(AnnyCharacter character)
        {
            if (character == null) throw new System.ArgumentNullException(nameof(character));

            character.mode = mode;
            character.singlePrecision = singlePrecision;
            character.phenotype = new List<AnnySliderEntry>(phenotype);

            AnnyMeshOptions options = character.meshOptions;
            options.MaxBoneInfluences = maxBoneInfluences;
            options.BuildBlendshapes = buildBlendshapes;
            options.ExpandUvCorners = expandUvCorners;
        }

        /// <summary>True when the character already holds exactly this configuration.</summary>
        public bool Matches(AnnyCharacter character)
        {
            if (character == null) return false;
            if (mode != character.mode || singlePrecision != character.singlePrecision) return false;
            if (phenotype.Count != character.phenotype.Count) return false;

            for (int i = 0; i < phenotype.Count; i++)
            {
                if (phenotype[i].name != character.phenotype[i].name) return false;
                if (phenotype[i].value != character.phenotype[i].value) return false;
            }

            AnnyMeshOptions options = character.meshOptions;
            return maxBoneInfluences == options.MaxBoneInfluences
                && buildBlendshapes == options.BuildBlendshapes
                && expandUvCorners == options.ExpandUvCorners;
        }
    }
}
