function direction = mass_tangent(gradient, r)
    normal = r / norm(r);
    direction = gradient - normal * (normal' * gradient);
end
