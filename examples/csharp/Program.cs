using AnnyExample;
using System.Text;
using System.Text.Json;
if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: AnnyExample /path/to/prepared-model.safetensors");
    return 2;
}
using var model = new AnnyModel(args[0]);
var mesh = model.Generate("{\"phenotype_kwargs\":{\"height\":0.6,\"weight\":0.4}}");
if (mesh.Vertices.Length == 0 || mesh.Faces.Length == 0) throw new InvalidDataException("Empty mesh.");
Console.WriteLine($"Generated {mesh.Vertices.Length / 3} vertices and {mesh.Faces.Length / mesh.FaceSize} faces in native Rust.");
using var measurements = JsonDocument.Parse(model.Query("{\"operation\":\"measure\"}"));
if (measurements.RootElement.GetProperty("height")[0].GetDouble() <= 0) throw new InvalidDataException("Invalid height.");
var glb = model.ExportGlb(optionsJson: "{\"rigged\":true}");
if (glb.Length < 12 || Encoding.ASCII.GetString(glb, 0, 4) != "glTF") throw new InvalidDataException("Invalid GLB.");
using var transformed = model.Transform("[{\"op\":\"triangulate\"},{\"op\":\"compact-skinning-weights\"}]");
using var pose = JsonDocument.Parse(model.TransferPoseTo(transformed));
_ = pose.RootElement.GetProperty("pose_parameters");
var prepared = transformed.SavePrepared();
var temporary = Path.Combine(Path.GetTempPath(), $"anny-smoke-{Guid.NewGuid():N}.safetensors");
try
{
    File.WriteAllBytes(temporary, prepared);
    using var reloaded = new AnnyModel(temporary);
    if (reloaded.Generate().Vertices.Length != mesh.Vertices.Length) throw new InvalidDataException("Reload changed vertex count.");
}
finally { File.Delete(temporary); }
Console.WriteLine("C# query, rigged GLB, transform, pose transfer and prepared reload: PASS");
using var single = model.ToSinglePrecision();
var singleMesh = single.Generate("{\"phenotype_kwargs\":{\"height\":0.6,\"weight\":0.4}}");
if (singleMesh.Vertices.Length != mesh.Vertices.Length || singleMesh.Vertices.Any(v => !float.IsFinite(v)))
    throw new InvalidDataException("Invalid f32 mesh.");
var error = mesh.Vertices.Zip(singleMesh.Vertices, (a,b) => Math.Abs(a-b)).Max();
if (error > 2e-4) throw new InvalidDataException($"f32 mismatch: {error}");
Console.WriteLine($"Native f32 C# evaluation: PASS; maximum vertex error {error:E}");
return 0;
