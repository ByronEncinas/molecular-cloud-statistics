from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Example values
simulation_name = "Magnetic Pockets"
num_particles = 5000
runtime = 123.45
average_density = 2.34e-21

# Create PDF
pdf_file = "simulation_report.pdf"
c = canvas.Canvas(pdf_file, pagesize=letter)
width, height = letter

# Title
c.setFont("Helvetica-Bold", 16)
c.drawString(72, height - 72, "Simulation Report")

# Body
c.setFont("Helvetica", 12)
lines = [
    f"Name:              {simulation_name}",
    f"Particles:         {num_particles:,}",
    f"Runtime:           {runtime:.2f} seconds",
    f"Average Density:   {average_density:.2e} g/cm³"
]

y = height - 120
for line in lines:
    c.drawString(72, y, line)
    y -= 20

c.save()
print(f"Report saved to {pdf_file}")