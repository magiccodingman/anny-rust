using System;
using System.IO;
using UnityEngine;

namespace Anny
{
    /// <summary>
    /// A prepared Anny model as a Unity asset. The payload is stored as a <c>.bytes</c> text asset
    /// so it survives import byte-for-byte and ships into player builds; the description is cached
    /// beside it so a runtime consumer can read topology without paying for a second parse.
    /// </summary>
    [CreateAssetMenu(fileName = "AnnyModel", menuName = "Anny/Model Asset")]
    public sealed class AnnyModelAsset : ScriptableObject
    {
        [Tooltip("The prepared model payload (.bytes), exactly as written by anny prepare.")]
        public TextAsset payload;

        [Tooltip("Load the runtime copy at single precision. The f64 model is the authority.")]
        public bool singlePrecision = true;

        [Tooltip("Cached JSON from anny_model_describe, so topology is available without a round trip.")]
        [TextArea(3, 12)]
        public string descriptionJson;

        private AnnyDescription parsed;

        /// <summary>Topology and labels, parsed from the cached description.</summary>
        public AnnyDescription Description
        {
            get
            {
                if (parsed == null && !string.IsNullOrEmpty(descriptionJson))
                {
                    parsed = JsonUtility.FromJson<AnnyDescription>(descriptionJson);
                }

                return parsed;
            }
        }

        /// <summary>Loads a double-precision model from the payload.</summary>
        public AnnyModel OpenWide()
        {
            return AnnyModel.FromBytes(PayloadBytes(), null);
        }

        /// <summary>
        /// Loads the runtime model. The prepared payload is the double-precision artefact, so the
        /// single-precision copy is converted from it; a caller that already prepared a 32-bit
        /// payload can load it directly with <see cref="AnnyModelF32.FromBytes"/> instead.
        /// </summary>
        public AnnyModelF32 OpenRuntime()
        {
            if (!singlePrecision)
            {
                throw new InvalidOperationException(
                    name + " is configured for the wide model only; clear singlePrecision or use OpenWide()");
            }

            using (AnnyModel wide = OpenWide())
            {
                return wide.ToSinglePrecision();
            }
        }

        /// <summary>Fills the cached description from a live model.</summary>
        public void CacheDescription(AnnyModel model)
        {
            descriptionJson = model.DescribeJson();
            parsed = null;
        }

        private byte[] PayloadBytes()
        {
            if (payload == null)
            {
                throw new InvalidOperationException(name + " has no payload assigned");
            }

            byte[] bytes = payload.bytes;
            if (bytes == null || bytes.Length == 0)
            {
                throw new InvalidOperationException(name + " has an empty payload");
            }

            return bytes;
        }
    }
}