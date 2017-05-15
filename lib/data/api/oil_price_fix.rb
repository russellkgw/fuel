class Data::Api::OilPriceFix
  def self.insert_averages
    missing_date = []
    oil_prices = OilPrice.where("date >= '1995-01-01'").order(date: :asc).all.select { |x| x.date.cwday <= 5 }

    (Date.new(1995, 1, 1)..(Date.current - 1.day)).each do |date|
      next if date.cwday >= 6
      missing_date << date if oil_prices.select { |x| x.date == date }.empty?
    end

    fixes = {}

    missing_date.each do |date|
      before = OilPrice.where("date < ?", date).order(date: :desc).first
      after = OilPrice.where("date > ?", date).order(date: :asc).first

      if before.present? && after.present?
        fixes[date.to_s] = (before.price + after.price) / 2.0
      end
    end

    fixes.each do |date, price|
      next if OilPrice.where(date: date).any?

      OilPrice.create({ currency: 'USD',
                        price: price,
                        date: date,
                        source: 'AVERAGE-https://www.quandl.com/api/v3/datasets/EIA/PET_RBRTE_D',
                        created_at: DateTime.now,
                        updated_at: DateTime.now })
    end

    puts missing_date.to_s
    puts fixes.to_s
  end
end