export function StatsSection() {
  const stats = [
    {
      value: "$50M+",
      label: "Fraud Prevented",
      description: "In fraudulent transactions blocked"
    },
    {
      value: "99.5%",
      label: "Detection Accuracy",
      description: "With minimal false positives"
    },
    {
      value: "2ms",
      label: "Response Time",
      description: "Real-time fraud detection"
    },
    {
      value: "500+",
      label: "Businesses Protected",
      description: "Across 50+ countries"
    }
  ]

  return (
    <section className="w-full py-16 px-5 relative">
      <div className="max-w-[1320px] mx-auto">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              <div className="text-3xl md:text-4xl font-bold text-primary mb-2">
                {stat.value}
              </div>
              <div className="text-lg font-semibold text-foreground mb-1">
                {stat.label}
              </div>
              <div className="text-sm text-muted-foreground">
                {stat.description}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
