export default function TestPage() {
  return (
    <div className="p-8 bg-red-500">
      <h1 className="text-4xl font-bold text-white mb-4">Tailwind Test</h1>
      <div className="bg-blue-500 p-4 rounded-lg shadow-lg">
        <p className="text-white">If you can see this styled content, Tailwind is working!</p>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-4">
        <div className="bg-green-500 p-4 rounded text-white">Green Card</div>
        <div className="bg-yellow-500 p-4 rounded text-black">Yellow Card</div>
      </div>
    </div>
  )
}
