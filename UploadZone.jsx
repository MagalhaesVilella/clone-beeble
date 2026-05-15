export default function UploadZone({ accept, label, file, previewUrl, onFile }) {
  return (
    <label className="upload-zone">
      <input
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          onFile(f || null);
        }}
      />
      <div className="upload-inner">
        {previewUrl ? (
          file?.type?.startsWith('video/') ? (
            <video className="preview" src={previewUrl} controls muted playsInline />
          ) : (
            <img className="preview" src={previewUrl} alt="" />
          )
        ) : (
          <span>{label}</span>
        )}
      </div>
      {file && <p className="fname">{file.name}</p>}
    </label>
  );
}
