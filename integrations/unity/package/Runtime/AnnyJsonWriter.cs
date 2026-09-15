using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace Anny
{
    /// <summary>
    /// Minimal JSON writer for values the native ABI accepts as parameters. Unity has no
    /// dictionary-capable serializer, and the native side rejects unknown fields, so the
    /// emitted keys are the exact field names of anny-core's <c>Parameters</c> struct.
    /// </summary>
    public static class AnnyJsonWriter
    {
        /// <summary>Appends a JSON string literal with the required escapes.</summary>
        public static void AppendString(StringBuilder builder, string value)
        {
            builder.Append('"');
            for (int i = 0; i < value.Length; i++)
            {
                char c = value[i];
                switch (c)
                {
                    case '"':
                        builder.Append("\\\"");
                        break;
                    case '\\':
                        builder.Append("\\\\");
                        break;
                    case '\n':
                        builder.Append("\\n");
                        break;
                    case '\r':
                        builder.Append("\\r");
                        break;
                    case '\t':
                        builder.Append("\\t");
                        break;
                    default:
                        if (c < ' ')
                        {
                            builder.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        }
                        else
                        {
                            builder.Append(c);
                        }

                        break;
                }
            }

            builder.Append('"');
        }

        /// <summary>
        /// Appends a number in round-trip ("R") form under the invariant culture so the value
        /// the native side parses is bit-identical to the value handed in.
        /// </summary>
        public static void AppendNumber(StringBuilder builder, double value)
        {
            if (double.IsNaN(value) || double.IsInfinity(value))
            {
                throw new System.ArgumentOutOfRangeException(
                    "value",
                    value,
                    "Anny parameters must be finite; NaN/Infinity cannot be expressed in JSON.");
            }

            builder.Append(value.ToString("R", CultureInfo.InvariantCulture));
        }

        /// <summary>Writes a <c>{"label": value}</c> object from a labelled scalar set.</summary>
        public static void AppendMap(StringBuilder builder, Dictionary<string, float> map)
        {
            builder.Append('{');
            bool first = true;
            foreach (KeyValuePair<string, float> entry in map)
            {
                if (!first)
                {
                    builder.Append(',');
                }

                first = false;
                AppendString(builder, entry.Key);
                builder.Append(':');
                AppendNumber(builder, entry.Value);
            }

            builder.Append('}');
        }

        /// <summary>Writes a tensor as nested arrays in row-major order.</summary>
        public static void AppendTensor(StringBuilder builder, AnnyTensor tensor)
        {
            int[] shape = tensor.Shape;
            for (int i = 0; i < shape.Length; i++)
            {
                builder.Append('[');
            }

            for (int i = 0; i < tensor.Length; i++)
            {
                if (i > 0)
                {
                    builder.Append(',');
                }

                AppendNumber(builder, tensor.Data[i]);
            }

            for (int i = 0; i < shape.Length; i++)
            {
                builder.Append(']');
            }
        }
    }
}