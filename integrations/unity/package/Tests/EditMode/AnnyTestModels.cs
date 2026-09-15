using System;
using System.IO;
using Anny;
using UnityEngine;

namespace Anny.Tests
{
    /// <summary>
    /// Locates the prepared model the Unity tests run against. The repository keeps prepared
    /// models in <c>output/</c>, which is not part of the package, so the path is discovered
    /// rather than assumed: <c>ANNY_MODEL</c> wins, then the repository location derived from the
    /// project folder, then any model already imported into the project.
    /// </summary>
    public static class AnnyTestModels
    {
        private static string cachedPath;

        /// <summary>Absolute path of the prepared f64 model used by the tests.</summary>
        public static string Path
        {
            get
            {
                if (cachedPath != null)
                {
                    return cachedPath;
                }

                string fromEnvironment = Environment.GetEnvironmentVariable("ANNY_MODEL");
                if (!string.IsNullOrEmpty(fromEnvironment) && File.Exists(fromEnvironment))
                {
                    cachedPath = fromEnvironment;
                    return cachedPath;
                }

                foreach (string candidate in Candidates())
                {
                    if (File.Exists(candidate))
                    {
                        cachedPath = candidate;
                        return cachedPath;
                    }
                }

                throw new FileNotFoundException(
                    "no prepared Anny model found. Set ANNY_MODEL to a prepared .safetensors file. " +
                    "Looked in: " + string.Join(", ", Candidates()));
            }
        }

        /// <summary>True when a model is available, so a caller can skip instead of failing.</summary>
        public static bool Available
        {
            get
            {
                try
                {
                    string unused = Path;
                    return unused.Length > 0;
                }
                catch (FileNotFoundException)
                {
                    return false;
                }
            }
        }

        /// <summary>Loads the prepared model.</summary>
        /// <summary>The prepared model's path, as the suites refer to it.</summary>
        public static string PreparedModelPath
        {
            get { return Path; }
        }

        /// <summary>Vertex count of the model the suites run against.</summary>
        public const int ExpectedVertices = 13718;

        /// <summary>Bone count of the model the suites run against.</summary>
        public const int ExpectedBones = 104;

        public static AnnyModel Load()
        {
            return AnnyModel.Load(Path);
        }

        /// <summary>Loads the prepared model at single precision.</summary>
        public static AnnyModelF32 LoadSingle()
        {
            using (AnnyModel wide = Load())
            {
                return wide.ToSinglePrecision();
            }
        }

        private static string[] Candidates()
        {
            string project = Application.dataPath;
            return new[]
            {
                // The repository keeps prepared models beside the crates, not in the package.
                System.IO.Path.GetFullPath(System.IO.Path.Combine(project, "../../../../output/ci-model.safetensors")),
                System.IO.Path.GetFullPath(System.IO.Path.Combine(project, "../../../output/ci-model.safetensors")),
                System.IO.Path.Combine(project, "AnnyModel/ci-model.safetensors"),
            };
        }
    }
}