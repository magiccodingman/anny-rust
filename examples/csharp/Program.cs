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
var edited = AnnyGltf.Edit(glb, "[{\"op\":\"set-material\",\"mesh\":0,\"primitive\":0,\"material\":{\"name\":\"managed-edit\",\"roughness\":0.4}}]");
using var gltfInfo = JsonDocument.Parse(AnnyGltf.Query(edited));
if (gltfInfo.RootElement.GetProperty("meshes").GetInt32() != 1) throw new InvalidDataException("glTF edit lost mesh.");
using var derivative = JsonDocument.Parse(model.Query("{\"operation\":\"jvp\",\"direction\":{\"phenotypes\":{\"height\":1.0}}}"));
_ = derivative.RootElement.GetProperty("vertices");
var tensorShape = new[] { 1, mesh.Vertices.Length / 3, 3 };
var cotangents = Enumerable.Repeat(1.0, mesh.Vertices.Length).ToArray();
var vjpRequest = JsonSerializer.Serialize(new
{
    operation = "vjp",
    selection = new { phenotypes = new[] { "height" } },
    cotangents = new { vertices = new { shape = tensorShape, data = cotangents } }
});
using var transpose = JsonDocument.Parse(model.Query(vjpRequest));
if (!transpose.RootElement.TryGetProperty("phenotypes", out var phenotypeGradient)
    || !phenotypeGradient.TryGetProperty("height", out _))
    throw new InvalidDataException("C# VJP did not return the selected phenotype gradient.");
var refineRequest = JsonSerializer.Serialize(new
{
    operation = "refine",
    target = new { shape = tensorShape, data = mesh.Vertices },
    options = new
    {
        steps = 1,
        learning_rate = 0.01,
        selection = new { phenotypes = new[] { "height" } }
    }
});
using var refined = JsonDocument.Parse(model.Query(refineRequest));
if (!refined.RootElement.TryGetProperty("losses", out var losses) || losses.GetArrayLength() < 2)
    throw new InvalidDataException("C# refinement did not execute an Adam step.");
Console.WriteLine("Native glTF edit/query plus JVP/VJP/refinement through C#: PASS");
return 0;
