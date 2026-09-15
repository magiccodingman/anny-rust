using System;
using System.Collections.Generic;
using UnityEngine;

namespace Anny
{
    /// <summary>
    /// Bakes native motion into ordinary Unity <see cref="AnimationClip"/> assets.
    ///
    /// Both entries walk a sequence of keyframes and record the rig's local transforms, so the clip is
    /// Unity-native: it plays through the Animator, it serialises as an asset, and it needs neither the
    /// native plugin nor Anny at runtime. <see cref="BakePhenotypeSequence"/> drives the full evaluation
    /// path (the sliders), while <see cref="BakePoseSequence"/> drives the character's pose session,
    /// which is the path that only pays for the pose-dependent half of the evaluation.
    /// </summary>
    public static class AnnyClipBaker
    {
        /// <summary>Bakes a slider sequence: each keyframe is a set of phenotype values.</summary>
        public static AnimationClip BakePhenotypeSequence(
            AnnyCharacter character,
            IList<IList<AnnySliderEntry>> keyframes,
            float frameRate)
        {
            if (character == null)
            {
                throw new ArgumentNullException("character");
            }

            if (keyframes == null || keyframes.Count == 0)
            {
                throw new ArgumentException("a clip needs at least one keyframe", "keyframes");
            }

            Rig(character);

            return Bake(character, keyframes.Count, frameRate, delegate(int frame)
            {
                IList<AnnySliderEntry> entries = keyframes[frame];
                for (int i = 0; i < entries.Count; i++)
                {
                    character.SetPhenotype(entries[i].name, entries[i].value);
                }

                character.Apply();
            });
        }

        /// <summary>
        /// Bakes a pose sequence through the character's pose session. Each keyframe is an Anny-space
        /// world pose in the form the native pose parser accepts, which is what
        /// <see cref="AnnySkeleton.CaptureAnnyWorldPose"/> produces — so a Unity-posed rig can be
        /// recorded straight back into a clip.
        /// </summary>
        public static AnimationClip BakePoseSequence(
            AnnyCharacter character,
            IList<Dictionary<string, Matrix4x4>> keyframes,
            float frameRate)
        {
            if (character == null)
            {
                throw new ArgumentNullException("character");
            }

            if (keyframes == null || keyframes.Count == 0)
            {
                throw new ArgumentException("a clip needs at least one keyframe", "keyframes");
            }

            Rig(character);

            return Bake(character, keyframes.Count, frameRate, delegate(int frame)
            {
                // AnnyCharacter.ApplyPose routes through the native pose session when there is one and
                // falls back to a full evaluation when there is not, so the clip is baked either way.
                character.ApplyPose(AnnyRepresentation.NamedPosesToJson(keyframes[frame]));
            });
        }

        private static AnnyRig Rig(AnnyCharacter character)
        {
            AnnyRig rig = character.Rig;
            if (rig == null || rig.Count == 0)
            {
                throw new InvalidOperationException("generate the character before baking a clip");
            }

            return rig;
        }

        private static AnimationClip Bake(AnnyCharacter character, int frames, float frameRate, Action<int> poseAt)
        {
            AnnyRig rig = character.Rig;
            float step = 1f / frameRate;

            AnimationCurve[] positionX = new AnimationCurve[rig.Count];
            AnimationCurve[] positionY = new AnimationCurve[rig.Count];
            AnimationCurve[] positionZ = new AnimationCurve[rig.Count];
            AnimationCurve[] rotationX = new AnimationCurve[rig.Count];
            AnimationCurve[] rotationY = new AnimationCurve[rig.Count];
            AnimationCurve[] rotationZ = new AnimationCurve[rig.Count];
            AnimationCurve[] rotationW = new AnimationCurve[rig.Count];

            for (int bone = 0; bone < rig.Count; bone++)
            {
                positionX[bone] = new AnimationCurve();
                positionY[bone] = new AnimationCurve();
                positionZ[bone] = new AnimationCurve();
                rotationX[bone] = new AnimationCurve();
                rotationY[bone] = new AnimationCurve();
                rotationZ[bone] = new AnimationCurve();
                rotationW[bone] = new AnimationCurve();
            }

            for (int frame = 0; frame < frames; frame++)
            {
                float time = frame * step;
                poseAt(frame);

                for (int bone = 0; bone < rig.Count; bone++)
                {
                    Transform transform = rig.Bones[bone];
                    if (transform == null)
                    {
                        continue;
                    }

                    Vector3 position = transform.localPosition;
                    Quaternion rotation = transform.localRotation;

                    positionX[bone].AddKey(time, position.x);
                    positionY[bone].AddKey(time, position.y);
                    positionZ[bone].AddKey(time, position.z);
                    rotationX[bone].AddKey(time, rotation.x);
                    rotationY[bone].AddKey(time, rotation.y);
                    rotationZ[bone].AddKey(time, rotation.z);
                    rotationW[bone].AddKey(time, rotation.w);
                }
            }

            AnimationClip clip = new AnimationClip();
            clip.frameRate = frameRate;
            clip.name = character.name + " (Anny)";

            for (int bone = 0; bone < rig.Count; bone++)
            {
                Transform transform = rig.Bones[bone];
                if (transform == null)
                {
                    continue;
                }

                string path = RelativePath(rig, bone);
                Curve(clip, path, "localPosition.x", positionX[bone]);
                Curve(clip, path, "localPosition.y", positionY[bone]);
                Curve(clip, path, "localPosition.z", positionZ[bone]);
                Curve(clip, path, "localRotation.x", rotationX[bone]);
                Curve(clip, path, "localRotation.y", rotationY[bone]);
                Curve(clip, path, "localRotation.z", rotationZ[bone]);
                Curve(clip, path, "localRotation.w", rotationW[bone]);
            }

            return clip;
        }

        private static void Curve(AnimationClip clip, string path, string property, AnimationCurve curve)
        {
            // Tangent modes live in UnityEditor and this code ships in players, so the curves are
            // built with the default auto tangents. Each baked key comes from a real native
            // evaluation, so there is nothing to smooth over.
            curve.preWrapMode = WrapMode.Clamp;
            curve.postWrapMode = WrapMode.Clamp;
            clip.SetCurve(path, typeof(Transform), property, curve);
        }

        /// <summary>
        /// Path of a bone relative to the rig root, which is the form <c>AnimationClip.SetCurve</c>
        /// wants. Bones are stored parents-before-children, so one pass suffices.
        /// </summary>
        private static string RelativePath(AnnyRig rig, int bone)
        {
            if (rig.Bones[bone] == rig.Root)
            {
                return string.Empty;
            }

            string path = rig.Bones[bone].name;
            int parent = rig.Parents == null ? -1 : rig.Parents[bone];
            int guard = 0;

            while (parent >= 0 && rig.Bones[parent] != rig.Root && guard++ < rig.Count)
            {
                path = rig.Bones[parent].name + "/" + path;
                parent = rig.Parents[parent];
            }

            return path;
        }
    }
}
