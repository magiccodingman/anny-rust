using AnnyExample;
if (args.Length != 1)
{
    Console.Error.WriteLine("Usage: AnnyExample /path/to/prepared-model.safetensors");
    return 2;
}
using var model = new AnnyModel(args[0]);
var mesh = model.Generate("{\"phenotype_kwargs\":{\"height\":0.6,\"weight\":0.4}}");
Console.WriteLine($"Generated {mesh.Vertices.Length / 3} vertices and {mesh.Faces.Length / mesh.FaceSize} faces in native Rust.");
return 0;
