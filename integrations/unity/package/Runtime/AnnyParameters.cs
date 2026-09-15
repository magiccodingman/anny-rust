using System.Collections.Generic;
using System.Text;

namespace Anny
{
    /// <summary>
    /// The parameter set one evaluation consumes: phenotype and local-change sliders, facial
    /// actions, and an optional pose tensor. Field names and accepted forms mirror anny-core's
    /// <c>Parameters</c> struct exactly, because the native parser rejects unknown fields.
    /// </summary>
    public sealed class AnnyParameters
    {
        /// <summary>Phenotype sliders, e.g. <c>gender</c>, <c>age</c>, <c>muscle</c>, <c>weight</c>, <c>height</c>, <c>proportions</c>.</summary>
        public readonly Dictionary<string, float> PhenotypeKwargs = new Dictionary<string, float>();

        /// <summary>Local-change sliders, e.g. <c>l_eye</c>, <c>nose</c>; labels come from the model.</summary>
        public readonly Dictionary<string, float> LocalChangesKwargs = new Dictionary<string, float>();

        /// <summary>Facial-action sliders; labels come from the model.</summary>
        public readonly Dictionary<string, float> FacialActions = new Dictionary<string, float>();

        /// <summary>
        /// Pose of shape <c>[batch, bones, 4, 4]</c> in the parameterization named by
        /// <see cref="PoseParameterization"/>. Null means the rest pose.
        /// </summary>
        public AnnyTensor PoseParameters { get; set; }

        /// <summary>One of <c>world</c>, <c>local_bone_world</c>, <c>local_bone</c>, <c>local_ref</c>, <c>world_orient</c>. Null uses the model default.</summary>
        public string PoseParameterization { get; set; }

        /// <summary>Asks the native side to also return bone end positions.</summary>
        public bool ReturnBoneEnds { get; set; }

        /// <summary>Rest (unposed) parameters: every slider is simply absent.</summary>
        public static AnnyParameters Rest()
        {
            return new AnnyParameters();
        }

        /// <summary>Sets one phenotype slider, creating it when absent.</summary>
        public AnnyParameters Phenotype(string label, float value)
        {
            PhenotypeKwargs[label] = value;
            return this;
        }

        /// <summary>Sets one local-change slider, creating it when absent.</summary>
        public AnnyParameters LocalChange(string label, float value)
        {
            LocalChangesKwargs[label] = value;
            return this;
        }

        /// <summary>Sets one facial action, creating it when absent.</summary>
        public AnnyParameters FacialAction(string label, float value)
        {
            FacialActions[label] = value;
            return this;
        }

        /// <summary>Attaches a pose tensor and its parameterization.</summary>
        public AnnyParameters WithPose(AnnyTensor pose, string parameterization = null)
        {
            PoseParameters = pose;
            PoseParameterization = parameterization;
            return this;
        }

        /// <summary>Independent copy, so a preset can be handed out without aliasing its sliders.</summary>
        public AnnyParameters Clone()
        {
            AnnyParameters copy = new AnnyParameters
            {
                PoseParameters = PoseParameters,
                PoseParameterization = PoseParameterization,
                ReturnBoneEnds = ReturnBoneEnds,
            };
            foreach (KeyValuePair<string, float> entry in PhenotypeKwargs)
            {
                copy.PhenotypeKwargs[entry.Key] = entry.Value;
            }

            foreach (KeyValuePair<string, float> entry in LocalChangesKwargs)
            {
                copy.LocalChangesKwargs[entry.Key] = entry.Value;
            }

            foreach (KeyValuePair<string, float> entry in FacialActions)
            {
                copy.FacialActions[entry.Key] = entry.Value;
            }

            return copy;
        }

        /// <summary>Builds the JSON string the native evaluation entry points take.</summary>
        public string ToJson()
        {
            StringBuilder builder = new StringBuilder(256);
            builder.Append("{\"phenotype_kwargs\":");
            AnnyJsonWriter.AppendMap(builder, PhenotypeKwargs);
            builder.Append(",\"local_changes_kwargs\":");
            AnnyJsonWriter.AppendMap(builder, LocalChangesKwargs);
            builder.Append(",\"facial_actions\":");
            AnnyJsonWriter.AppendMap(builder, FacialActions);
            builder.Append(",\"pose_parameters\":");
            if (PoseParameters == null)
            {
                builder.Append("null");
            }
            else
            {
                AnnyJsonWriter.AppendTensor(builder, PoseParameters);
            }

            if (PoseParameterization != null)
            {
                builder.Append(",\"pose_parameterization\":");
                AnnyJsonWriter.AppendString(builder, PoseParameterization);
            }

            builder.Append(",\"return_bone_ends\":");
            builder.Append(ReturnBoneEnds ? "true" : "false");
            builder.Append('}');
            return builder.ToString();
        }

        public override string ToString()
        {
            return ToJson();
        }
    }
}