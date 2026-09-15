namespace Anny
{
    /// <summary>
    /// Anything that can hand back a named tensor from a recent evaluation: an owning
    /// <see cref="AnnyOutputF32"/>, or a pose session, whose result lives inside the session and
    /// must not be disposed by the consumer.
    /// <para>
    /// The distinction matters for lifetime, not for the values: the session path skips the
    /// coefficient and posture work that the full evaluation repeats.
    /// </para>
    /// </summary>
    public interface IAnnyEvaluation
    {
        AnnyTensorF32 Tensor(string name);

        bool HasTensor(string name);
    }
}